"""
AstroNifty RealtimeHub — Central real-time data distribution hub (Multi-User).

ONE WebSocket connection to Kite for market data (shared across all users).
Uses the first authenticated user's token. If that user's session expires,
automatically switches to another authenticated user's token.

Tick data is SHARED — all users see the same prices.
Per-user position monitoring happens via signal_bus callbacks.

Tick flow:
    KiteTicker -> on_tick() -> update cache -> rebuild chain/greeks
    -> push to dashboard WS -> publish on SignalBus
    -> per-user position monitoring via callbacks
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytz
from kiteconnect import KiteTicker
from loguru import logger
from scipy.stats import norm

if TYPE_CHECKING:
    from core.signal_bus import SignalBus
    from core.user_manager import UserManager

from config import (
    INDICES,
    SECTORS,
    KITE_API_KEY,
)

IST = pytz.timezone("Asia/Kolkata")

# ── Greeks constants ─────────────────────────────────────────────────────
RISK_FREE_RATE = 0.07
TRADING_DAYS = 252


def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes d1 component."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> dict:
    """Compute Black-Scholes greeks for a single option."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": sigma}

    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    sqrt_T = math.sqrt(T)

    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    gamma = n_d1 / (S * sigma * sqrt_T)
    vega = S * n_d1 * sqrt_T / 100

    if opt_type == "CE":
        delta = N_d1
        theta = (-(S * n_d1 * sigma) / (2 * sqrt_T) - r * K * math.exp(-r * T) * N_d2) / TRADING_DAYS
    else:
        delta = N_d1 - 1
        theta = (-(S * n_d1 * sigma) / (2 * sqrt_T) + r * K * math.exp(-r * T) * norm.cdf(-d2)) / TRADING_DAYS

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 2),
        "vega": round(vega, 2),
        "iv": round(sigma * 100, 2),
    }


def _implied_vol(price: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    """Newton-Raphson implied volatility solver."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0

    sigma = 0.3
    for _ in range(50):
        d1 = _bs_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        if opt_type == "CE":
            theo = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            theo = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        vega = S * norm.pdf(d1) * math.sqrt(T)
        if vega < 1e-10:
            break

        diff = theo - price
        sigma -= diff / vega

        if sigma < 0.01:
            sigma = 0.01
        if sigma > 5.0:
            sigma = 5.0

        if abs(diff) < 0.01:
            break

    return max(sigma, 0.01)


class RealtimeHub:
    """
    Central hub — ONE WebSocket connection shared across all users.

    Market data (prices, OI, greeks) is SHARED. Every user sees the same ticks.
    The ticker connects using any authenticated user's Kite access token.
    If the active user's token expires, the hub automatically switches to
    another authenticated user's token.

    Attributes:
        latest_prices: {instrument_token: {ltp, oi, volume, bid, ask, ...}}
        latest_chains: {index_name: DataFrame} — rebuilt on each OI tick
        latest_greeks: {index_name: {strike: {ce_greeks, pe_greeks}}}
        tick_count: Total ticks received since start
        subscribed_tokens: Currently subscribed instrument tokens
    """

    def __init__(
        self,
        signal_bus: "SignalBus",
        user_manager: "UserManager",
        dashboard_ws=None,
    ) -> None:
        self.signal_bus = signal_bus
        self.user_manager = user_manager
        self.dashboard_ws = dashboard_ws

        # ── Active ticker user tracking ───────────────────────────
        self._active_user_id: Optional[str] = None
        self._active_kite_client = None

        # ── Caches ────────────────────────────────────────────────
        self.latest_prices: Dict[int, dict] = {}
        self.latest_chains: Dict[str, pd.DataFrame] = {}
        self.latest_greeks: Dict[str, dict] = {}
        self.previous_chains: Dict[str, pd.DataFrame] = {}
        self.tick_count: int = 0
        self.subscribed_tokens: List[int] = []
        self.last_full_analysis_ts: float = 0.0

        # ── Token mapping ─────────────────────────────────────────
        self._token_to_symbol: Dict[int, str] = {}
        self._symbol_to_token: Dict[str, int] = {}
        self._token_to_index: Dict[int, str] = {}
        self._token_to_strike: Dict[int, float] = {}
        self._token_to_type: Dict[int, str] = {}
        self._spot_tokens: Dict[str, int] = {}
        self._sector_tokens: Dict[str, int] = {}

        # ── Throttle timestamps for dashboard push ────────────────
        self._last_push: Dict[str, float] = defaultdict(float)

        # ── Internal ticker ───────────────────────────────────────
        self._ticker: Optional[KiteTicker] = None
        self._ticker_thread: Optional[threading.Thread] = None
        self._connected = threading.Event()
        self._running = False

        # ── Callbacks ─────────────────────────────────────────────
        self._on_tick_callbacks: List[Callable] = []

        # ── Lock for thread safety ────────────────────────────────
        self._lock = threading.Lock()

        logger.info("RealtimeHub initialised (multi-user, shared ticker)")

    # ------------------------------------------------------------------
    # Active Kite client management
    # ------------------------------------------------------------------
    def get_active_kite_client(self):
        """Return the currently active Kite client for API calls.

        If the active user's session is no longer valid, switch to
        another authenticated user.
        """
        # Check if current active client is still valid
        if self._active_user_id and self._active_kite_client:
            session = self.user_manager.get_session(self._active_user_id)
            if session and session.is_authenticated:
                return self._active_kite_client

        # Active user expired — find a new one
        return self._switch_to_new_user()

    def _switch_to_new_user(self):
        """Switch the ticker connection to a new authenticated user's token."""
        sessions = self.user_manager.get_all_authenticated_sessions()
        if not sessions:
            logger.warning("No authenticated users available for ticker connection")
            self._active_user_id = None
            self._active_kite_client = None
            return None

        # Pick the first available authenticated user
        new_session = sessions[0]
        old_user = self._active_user_id
        self._active_user_id = new_session.user_id
        self._active_kite_client = new_session.kite_client

        if old_user and old_user != self._active_user_id:
            logger.warning(
                "Ticker connection switched: user {} -> user {}",
                old_user,
                self._active_user_id,
            )
            # Reconnect the ticker with the new user's token
            self._reconnect_ticker()

        return self._active_kite_client

    def _reconnect_ticker(self) -> None:
        """Reconnect the KiteTicker with the new active user's access token."""
        if not self._running:
            return

        logger.info("Reconnecting ticker with new user token...")

        # Close existing ticker
        if self._ticker:
            try:
                self._ticker.close()
            except Exception:
                pass
            self._ticker = None

        self._connected.clear()

        # Start fresh with new token
        if self._active_kite_client:
            try:
                access_token = self._active_kite_client.kite.access_token
                if not access_token:
                    logger.error("New active user has no access token")
                    return

                self._ticker = KiteTicker(KITE_API_KEY, access_token)
                self._ticker.on_ticks = self._on_ticks
                self._ticker.on_connect = self._on_connect
                self._ticker.on_close = self._on_close
                self._ticker.on_error = self._on_error
                self._ticker.on_reconnect = self._on_reconnect

                self._ticker.connect(threaded=True)
                logger.info("Ticker reconnected with user {}", self._active_user_id)
            except Exception as e:
                logger.exception("Ticker reconnection failed: {}", e)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------
    def register_tick_callback(self, callback: Callable) -> None:
        """Register an additional callback to run on every tick batch."""
        self._on_tick_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Subscribe to ALL relevant instrument tokens via KiteTicker.

        Uses the first authenticated user's Kite access token.
        When that user's session expires, automatically switches to
        another authenticated user.
        """
        logger.info("=" * 50)
        logger.info("REALTIME HUB — STARTING (SHARED TICKER)")
        logger.info("=" * 50)

        self._running = True

        # Find an authenticated user to connect with
        kite_client = self.get_active_kite_client()
        if not kite_client:
            logger.error("No authenticated users — RealtimeHub cannot start ticker")
            logger.info("RealtimeHub will start ticker when first user authenticates")
            return

        self._start_ticker_with_client(kite_client)

    def start_if_not_running(self) -> None:
        """Called when a new user authenticates — start ticker if not already running."""
        if self._connected.is_set():
            return  # Already running
        if not self._running:
            self._running = True

        kite_client = self.get_active_kite_client()
        if kite_client and not self._connected.is_set():
            logger.info("First user authenticated — starting ticker")
            self._start_ticker_with_client(kite_client)

    def _start_ticker_with_client(self, kite_client) -> None:
        """Internal: start the KiteTicker with a specific Kite client."""
        # Build token subscription list
        tokens = self._build_subscription_list(kite_client)
        if not tokens:
            logger.error("No tokens to subscribe — RealtimeHub cannot start")
            return

        self.subscribed_tokens = tokens
        logger.info("Subscribing to {} instrument tokens", len(tokens))

        # Connect KiteTicker using active user's token
        access_token = kite_client.kite.access_token
        self._ticker = KiteTicker(KITE_API_KEY, access_token)

        self._ticker.on_ticks = self._on_ticks
        self._ticker.on_connect = self._on_connect
        self._ticker.on_close = self._on_close
        self._ticker.on_error = self._on_error
        self._ticker.on_reconnect = self._on_reconnect

        self._ticker.connect(threaded=True)
        logger.info(
            "KiteTicker thread started (using user {}) — waiting for connection",
            self._active_user_id,
        )

    def stop(self) -> None:
        """Gracefully stop the ticker and clean up."""
        logger.info("RealtimeHub stopping...")
        self._running = False
        if self._ticker:
            try:
                self._ticker.close()
            except Exception as e:
                logger.error("Error closing ticker: {}", e)
            self._ticker = None
        self._active_user_id = None
        self._active_kite_client = None
        logger.info("RealtimeHub stopped")

    # ------------------------------------------------------------------
    # Subscription building
    # ------------------------------------------------------------------
    def _build_subscription_list(self, kite_client) -> List[int]:
        """Build the complete list of instrument tokens to subscribe."""
        tokens = []

        try:
            instruments = kite_client._get_instruments("NFO")
            nse_instruments = kite_client._get_instruments("NSE")
        except Exception as e:
            logger.error("Failed to fetch instruments: {}", e)
            return tokens

        # ── Spot index tokens ─────────────────────────────────────
        for idx_name, idx_config in INDICES.items():
            spot_sym = idx_config["symbol"]
            for inst in nse_instruments:
                if f"NSE:{inst['tradingsymbol']}" == spot_sym or inst["tradingsymbol"] == spot_sym.replace("NSE:", ""):
                    token = inst["instrument_token"]
                    self._spot_tokens[idx_name] = token
                    self._token_to_symbol[token] = spot_sym
                    self._symbol_to_token[spot_sym] = token
                    tokens.append(token)
                    logger.debug("Spot token: {} -> {}", idx_name, token)
                    break

        # ── Option chain tokens (ATM +/- 10) ─────────────────────
        for idx_name, idx_config in INDICES.items():
            if idx_config.get("is_gift"):
                continue

            strike_gap = idx_config["strike_gap"]
            atm_range = idx_config.get("atm_range", 10)

            spot_price = self._get_spot_price_for_index(idx_name, kite_client)
            if not spot_price:
                logger.warning("Cannot determine spot for {} — skipping chain tokens", idx_name)
                continue

            atm_strike = round(spot_price / strike_gap) * strike_gap
            strikes = [atm_strike + (i * strike_gap) for i in range(-atm_range, atm_range + 1)]

            from datetime import date as dt_date
            today = datetime.now(IST).date()
            idx_instruments = [
                i for i in instruments
                if i["name"] == idx_name
                and i["instrument_type"] in ("CE", "PE")
                and i["expiry"] >= today
            ]
            if not idx_instruments:
                logger.warning("No option instruments found for {}", idx_name)
                continue

            nearest_expiry = min(i["expiry"] for i in idx_instruments)
            logger.info("{} nearest expiry: {}", idx_name, nearest_expiry)

            for inst in idx_instruments:
                if inst["expiry"] != nearest_expiry:
                    continue
                if inst["strike"] not in strikes:
                    continue

                token = inst["instrument_token"]
                tokens.append(token)
                self._token_to_symbol[token] = f"NFO:{inst['tradingsymbol']}"
                self._symbol_to_token[f"NFO:{inst['tradingsymbol']}"] = token
                self._token_to_index[token] = idx_name
                self._token_to_strike[token] = inst["strike"]
                self._token_to_type[token] = inst["instrument_type"]

            logger.info(
                "{}: {} option tokens subscribed (ATM={}, range={})",
                idx_name,
                sum(1 for t in tokens if self._token_to_index.get(t) == idx_name),
                atm_strike,
                atm_range,
            )

        # ── Sector tokens ─────────────────────────────────────────
        for sector_name, sector_config in SECTORS.items():
            sector_sym = sector_config["symbol"]
            for inst in nse_instruments:
                full_sym = f"NSE:{inst['tradingsymbol']}"
                if full_sym == sector_sym:
                    token = inst["instrument_token"]
                    self._sector_tokens[sector_name] = token
                    self._token_to_symbol[token] = sector_sym
                    self._symbol_to_token[sector_sym] = token
                    tokens.append(token)
                    break

        logger.info("Total subscription: {} tokens (spot={}, sectors={}, options={})",
                     len(tokens),
                     len(self._spot_tokens),
                     len(self._sector_tokens),
                     len(self._token_to_index))

        return list(set(tokens))

    def _get_spot_price_for_index(self, idx_name: str, kite_client=None) -> Optional[float]:
        """Get current spot price for ATM calculation."""
        client = kite_client or self.get_active_kite_client()
        if not client:
            return None
        try:
            sym = INDICES[idx_name]["symbol"]
            ltp_data = client.get_ltp([sym])
            return list(ltp_data.values())[0]
        except Exception as e:
            logger.error("Failed to get spot price for {}: {}", idx_name, e)
            return None

    # ------------------------------------------------------------------
    # KiteTicker callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, ws, response) -> None:
        """Called when ticker connects — subscribe to all tokens."""
        logger.info("KiteTicker CONNECTED — subscribing {} tokens", len(self.subscribed_tokens))
        if self.subscribed_tokens:
            ws.subscribe(self.subscribed_tokens)
            ws.set_mode(ws.MODE_FULL, self.subscribed_tokens)
        self._connected.set()

    def _on_close(self, ws, code, reason) -> None:
        logger.warning("KiteTicker CLOSED: code={} reason={}", code, reason)
        self._connected.clear()

        # If the close was due to auth failure, try switching users
        if code in (4001, 4002, 403):
            logger.warning("Auth-related close — attempting user switch")
            new_client = self._switch_to_new_user()
            if new_client:
                logger.info("Will reconnect with new user on next reconnect attempt")

    def _on_error(self, ws, code, reason) -> None:
        logger.error("KiteTicker ERROR: code={} reason={}", code, reason)

    def _on_reconnect(self, ws, attempts_count) -> None:
        logger.info("KiteTicker reconnecting... attempt #{}", attempts_count)

        # On reconnect, check if current user is still valid
        if self._active_user_id:
            session = self.user_manager.get_session(self._active_user_id)
            if not session or not session.is_authenticated:
                logger.warning(
                    "Active user {} no longer authenticated — switching",
                    self._active_user_id,
                )
                new_client = self._switch_to_new_user()
                if new_client:
                    # Update the ticker's access token for reconnection
                    try:
                        access_token = new_client.kite.access_token
                        ws.access_token = access_token
                        logger.info(
                            "Updated ticker token to user {} for reconnect",
                            self._active_user_id,
                        )
                    except Exception:
                        logger.exception("Failed to update token on reconnect")

    # ------------------------------------------------------------------
    # THE CORE: on_ticks handler
    # ------------------------------------------------------------------
    def _on_ticks(self, ws, ticks: list) -> None:
        """Called every ~1 second with batch of tick data.

        Flow:
        1. Update latest_prices cache (SHARED)
        2. Check if OI changed -> rebuild option chain
        3. Recalculate greeks for changed strikes
        4. Push live prices to dashboard WS
        5. Publish tick_update on SignalBus
        6. Fire registered callbacks (includes per-user position monitoring)
        7. Every 60s, signal for full analysis cycle
        """
        if not ticks:
            return

        now = time.time()
        self.tick_count += len(ticks)
        oi_changed_indices = set()

        # ── 1. Update price cache (SHARED — all users see same prices) ─
        with self._lock:
            for tick in ticks:
                token = tick.get("instrument_token")
                if token is None:
                    continue

                prev = self.latest_prices.get(token, {})
                prev_oi = prev.get("oi", 0)

                self.latest_prices[token] = {
                    "ltp": tick.get("last_price", 0),
                    "oi": tick.get("oi", 0),
                    "volume": tick.get("volume_traded", 0),
                    "bid": tick.get("depth", {}).get("buy", [{}])[0].get("price", 0) if tick.get("depth") else 0,
                    "ask": tick.get("depth", {}).get("sell", [{}])[0].get("price", 0) if tick.get("depth") else 0,
                    "high": tick.get("ohlc", {}).get("high", 0),
                    "low": tick.get("ohlc", {}).get("low", 0),
                    "open": tick.get("ohlc", {}).get("open", 0),
                    "close": tick.get("ohlc", {}).get("close", 0),
                    "change": tick.get("change", 0),
                    "timestamp": datetime.now(IST).isoformat(),
                }

                new_oi = tick.get("oi", 0)
                if token in self._token_to_index and new_oi != prev_oi:
                    oi_changed_indices.add(self._token_to_index[token])

        # ── 2. Rebuild chains if OI changed ──────────────────────
        for idx_name in oi_changed_indices:
            self._rebuild_chain(idx_name)

        # ── 3. Recalculate greeks ────────────────────────────────
        for idx_name in oi_changed_indices:
            self._recalculate_greeks(idx_name)

        # ── 4. Push to dashboard via WebSocket ───────────────────
        self._push_live_prices(now)
        if oi_changed_indices:
            self._push_chain_update(now, oi_changed_indices)
            self._push_greeks_update(now, oi_changed_indices)
        self._push_sector_pulse(now)

        # ── 5. Publish on signal bus ─────────────────────────────
        try:
            self.signal_bus.publish("tick_update", {
                "tick_count": self.tick_count,
                "tokens_updated": len(ticks),
                "oi_changed": list(oi_changed_indices),
            })
        except Exception:
            pass

        # ── 6. Fire registered callbacks (per-user monitoring) ───
        for cb in self._on_tick_callbacks:
            try:
                cb(ticks)
            except Exception as e:
                logger.error("Tick callback error: {}", e)

        # ── 7. Every 60s trigger full analysis ───────────────────
        if now - self.last_full_analysis_ts >= 60.0:
            self.last_full_analysis_ts = now
            try:
                self.signal_bus.publish("oi_update", {
                    "source": "realtime_hub",
                    "indices": list(self.latest_chains.keys()),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Chain rebuilding from live tick cache
    # ------------------------------------------------------------------
    def _rebuild_chain(self, index_name: str) -> None:
        """Rebuild the option chain DataFrame for an index from live tick cache."""
        if index_name in self.latest_chains:
            self.previous_chains[index_name] = self.latest_chains[index_name].copy()

        rows = []
        strikes_seen = set()

        with self._lock:
            for token, data in self.latest_prices.items():
                if self._token_to_index.get(token) != index_name:
                    continue
                strike = self._token_to_strike.get(token)
                opt_type = self._token_to_type.get(token)
                if strike is None or opt_type is None:
                    continue
                strikes_seen.add(strike)

        for strike in sorted(strikes_seen):
            ce_data = {}
            pe_data = {}

            with self._lock:
                for token, data in self.latest_prices.items():
                    if (self._token_to_index.get(token) == index_name
                            and self._token_to_strike.get(token) == strike):
                        if self._token_to_type.get(token) == "CE":
                            ce_data = data
                            ce_data["token"] = token
                        elif self._token_to_type.get(token) == "PE":
                            pe_data = data
                            pe_data["token"] = token

            ce_oi = ce_data.get("oi", 0)
            pe_oi = pe_data.get("oi", 0)
            pcr = round(pe_oi / ce_oi, 4) if ce_oi else 0.0

            rows.append({
                "strike": strike,
                "ce_token": ce_data.get("token"),
                "ce_ltp": ce_data.get("ltp", 0),
                "ce_oi": ce_oi,
                "ce_volume": ce_data.get("volume", 0),
                "ce_iv": 0,
                "ce_bid": ce_data.get("bid", 0),
                "ce_ask": ce_data.get("ask", 0),
                "pe_token": pe_data.get("token"),
                "pe_ltp": pe_data.get("ltp", 0),
                "pe_oi": pe_oi,
                "pe_volume": pe_data.get("volume", 0),
                "pe_iv": 0,
                "pe_bid": pe_data.get("bid", 0),
                "pe_ask": pe_data.get("ask", 0),
                "pcr": pcr,
            })

        if rows:
            self.latest_chains[index_name] = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Greeks recalculation
    # ------------------------------------------------------------------
    def _recalculate_greeks(self, index_name: str) -> None:
        """Recalculate real-time greeks for all strikes of an index."""
        chain = self.latest_chains.get(index_name)
        if chain is None or chain.empty:
            return

        spot = self.get_live_price_by_index(index_name)
        if not spot or spot <= 0:
            return

        now = datetime.now(IST)
        days_to_expiry = max(1, 5)
        T = days_to_expiry / TRADING_DAYS

        greeks_data = {}
        for _, row in chain.iterrows():
            strike = row["strike"]
            strike_greeks = {}

            for opt_type, prefix in [("CE", "ce_"), ("PE", "pe_")]:
                ltp = row.get(f"{prefix}ltp", 0)
                if ltp > 0:
                    iv = _implied_vol(ltp, spot, strike, T, RISK_FREE_RATE, opt_type)
                    g = _bs_greeks(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                else:
                    g = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "iv": 0}

                strike_greeks[f"{prefix}delta"] = g["delta"]
                strike_greeks[f"{prefix}gamma"] = g["gamma"]
                strike_greeks[f"{prefix}theta"] = g["theta"]
                strike_greeks[f"{prefix}vega"] = g["vega"]
                strike_greeks[f"{prefix}iv"] = g["iv"]

            greeks_data[strike] = strike_greeks

            if index_name in self.latest_chains:
                mask = self.latest_chains[index_name]["strike"] == strike
                if mask.any():
                    self.latest_chains[index_name].loc[mask, "ce_iv"] = strike_greeks.get("ce_iv", 0)
                    self.latest_chains[index_name].loc[mask, "pe_iv"] = strike_greeks.get("pe_iv", 0)

        self.latest_greeks[index_name] = greeks_data

    # ------------------------------------------------------------------
    # Dashboard push methods (throttled)
    # ------------------------------------------------------------------
    def _push_live_prices(self, now: float) -> None:
        """Push live LTP to dashboard every 1 second."""
        if now - self._last_push.get("prices", 0) < 1.0:
            return
        self._last_push["prices"] = now

        if not self.dashboard_ws:
            return

        prices_payload = {}
        for idx_name, token in self._spot_tokens.items():
            data = self.latest_prices.get(token, {})
            if data:
                prices_payload[idx_name] = {
                    "ltp": data.get("ltp", 0),
                    "change": data.get("change", 0),
                    "high": data.get("high", 0),
                    "low": data.get("low", 0),
                    "open": data.get("open", 0),
                    "close": data.get("close", 0),
                    "volume": data.get("volume", 0),
                }

        self._broadcast_async({
            "channel": "prices",
            "data": prices_payload,
            "tick_count": self.tick_count,
            "ts": datetime.now(IST).isoformat(),
        })

    def _push_chain_update(self, now: float, changed_indices: set) -> None:
        """Push OI chain updates every 5 seconds."""
        if now - self._last_push.get("chain", 0) < 5.0:
            return
        self._last_push["chain"] = now

        if not self.dashboard_ws:
            return

        for idx_name in changed_indices:
            chain = self.latest_chains.get(idx_name)
            if chain is not None and not chain.empty:
                self._broadcast_async({
                    "channel": "chain",
                    "index": idx_name,
                    "data": chain.to_dict(orient="records"),
                    "ts": datetime.now(IST).isoformat(),
                })

    def _push_greeks_update(self, now: float, changed_indices: set) -> None:
        """Push real-time greeks every 5 seconds."""
        if now - self._last_push.get("greeks", 0) < 5.0:
            return
        self._last_push["greeks"] = now

        if not self.dashboard_ws:
            return

        for idx_name in changed_indices:
            greeks = self.latest_greeks.get(idx_name)
            if greeks:
                self._broadcast_async({
                    "channel": "greeks",
                    "index": idx_name,
                    "data": greeks,
                    "ts": datetime.now(IST).isoformat(),
                })

    def _push_sector_pulse(self, now: float) -> None:
        """Push sector data every 10 seconds."""
        if now - self._last_push.get("sectors", 0) < 10.0:
            return
        self._last_push["sectors"] = now

        if not self.dashboard_ws:
            return

        sector_data = self.get_sector_pulse()
        if sector_data:
            self._broadcast_async({
                "channel": "sectors",
                "data": sector_data,
                "ts": datetime.now(IST).isoformat(),
            })

    def _broadcast_async(self, payload: dict) -> None:
        """Non-blocking broadcast to dashboard WS."""
        if self.dashboard_ws:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self.dashboard_ws.broadcast(payload))
                else:
                    loop.run_until_complete(self.dashboard_ws.broadcast(payload))
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.dashboard_ws.broadcast(payload))

    # ------------------------------------------------------------------
    # Public API — used by engine and other modules
    # ------------------------------------------------------------------
    def get_live_chain(self, index: str) -> pd.DataFrame:
        """Return latest real-time option chain for an index (SHARED)."""
        chain = self.latest_chains.get(index)
        if chain is not None and not chain.empty:
            return chain.copy()
        return pd.DataFrame()

    def get_previous_chain(self, index: str) -> Optional[pd.DataFrame]:
        """Return previous cycle's chain for delta comparison."""
        prev = self.previous_chains.get(index)
        if prev is not None and not prev.empty:
            return prev.copy()
        return None

    def get_live_price(self, symbol: str) -> float:
        """Return latest LTP from cache by exchange:symbol (SHARED)."""
        token = self._symbol_to_token.get(symbol)
        if token and token in self.latest_prices:
            return self.latest_prices[token].get("ltp", 0.0)
        return 0.0

    def get_live_price_by_index(self, index: str) -> float:
        """Return latest spot price for an index name (SHARED)."""
        token = self._spot_tokens.get(index)
        if token and token in self.latest_prices:
            return self.latest_prices[token].get("ltp", 0.0)
        return 0.0

    def get_live_greeks(self, index: str) -> dict:
        """Return real-time greeks for all strikes of an index (SHARED)."""
        return self.latest_greeks.get(index, {})

    def get_sector_pulse(self) -> dict:
        """Real-time sector data from latest ticks (SHARED)."""
        result = {}
        for sector_name, token in self._sector_tokens.items():
            data = self.latest_prices.get(token, {})
            if not data:
                continue
            ltp = data.get("ltp", 0)
            prev_close = data.get("close", ltp)
            change = round(ltp - prev_close, 2) if prev_close else 0
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
            result[sector_name] = {
                "symbol": SECTORS[sector_name]["symbol"],
                "ltp": ltp,
                "change": change,
                "change_pct": change_pct,
                "volume": data.get("volume", 0),
            }
        return result

    def get_all_live_data(self) -> dict:
        """Return a complete snapshot of all live data for dashboard init."""
        return {
            "prices": {
                idx: {
                    "ltp": self.latest_prices.get(tok, {}).get("ltp", 0),
                    "change": self.latest_prices.get(tok, {}).get("change", 0),
                    "high": self.latest_prices.get(tok, {}).get("high", 0),
                    "low": self.latest_prices.get(tok, {}).get("low", 0),
                }
                for idx, tok in self._spot_tokens.items()
            },
            "chains": {
                idx: chain.to_dict(orient="records")
                for idx, chain in self.latest_chains.items()
                if not chain.empty
            },
            "greeks": self.latest_greeks,
            "sectors": self.get_sector_pulse(),
            "tick_count": self.tick_count,
            "connected": self._connected.is_set(),
            "subscribed_count": len(self.subscribed_tokens),
            "active_ticker_user": self._active_user_id,
        }

    # ------------------------------------------------------------------
    # Resubscription (ATM shift / expiry change)
    # ------------------------------------------------------------------
    def refresh_subscriptions(self) -> None:
        """Called when expiry changes or ATM shifts significantly."""
        logger.info("Refreshing RealtimeHub subscriptions...")

        if self._ticker and self._connected.is_set():
            try:
                self._ticker.unsubscribe(self.subscribed_tokens)
            except Exception as e:
                logger.error("Unsubscribe error: {}", e)

        # Clear option token mappings (keep spot/sector)
        self._token_to_index.clear()
        self._token_to_strike.clear()
        self._token_to_type.clear()

        # Rebuild using active kite client
        kite_client = self.get_active_kite_client()
        if kite_client:
            self.subscribed_tokens = self._build_subscription_list(kite_client)

            if self._ticker and self._connected.is_set() and self.subscribed_tokens:
                self._ticker.subscribe(self.subscribed_tokens)
                self._ticker.set_mode(self._ticker.MODE_FULL, self.subscribed_tokens)
                logger.info("Resubscribed to {} tokens", len(self.subscribed_tokens))
        else:
            logger.warning("No active kite client for resubscription")

    # ------------------------------------------------------------------
    # Health / status
    # ------------------------------------------------------------------
    def health_check(self) -> dict:
        return {
            "status": "green" if self._connected.is_set() else "red",
            "connected": self._connected.is_set(),
            "tick_count": self.tick_count,
            "subscribed_tokens": len(self.subscribed_tokens),
            "cached_prices": len(self.latest_prices),
            "chains_built": list(self.latest_chains.keys()),
            "active_ticker_user": self._active_user_id,
        }

    def __repr__(self) -> str:
        return (
            f"<RealtimeHub connected={self._connected.is_set()} "
            f"ticks={self.tick_count} "
            f"tokens={len(self.subscribed_tokens)} "
            f"active_user={self._active_user_id}>"
        )
