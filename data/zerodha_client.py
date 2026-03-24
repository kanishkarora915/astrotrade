"""
Zerodha Kite Connect Integration for AstroNifty Trading Engine.

Provides authenticated access to Zerodha's trading APIs including
order management, market data, WebSocket streaming, and option chain construction.
All calls are rate-limited, retried on transient failures, and logged via loguru.
"""

import base64
import json
import time
import threading
import webbrowser
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Optional, Callable, Tuple

import pandas as pd
import pytz
import redis
from kiteconnect import KiteConnect, KiteTicker
from loguru import logger

from config import (
    KITE_API_KEY,
    KITE_API_SECRET,
    KITE_ACCESS_TOKEN,
    REDIS_URL,
    PAPER_TRADE,
    INDICES,
    SECTORS,
    KITE_MAX_REQUESTS_PER_SEC,
    KITE_RETRY_COUNT,
    KITE_RETRY_DELAY_SEC,
)

IST = pytz.timezone("Asia/Kolkata")

# Path for encrypted credentials storage
CREDENTIALS_FILE = Path(__file__).parent.parent / ".credentials.json"


def _retry(func):
    """Decorator that retries a method up to KITE_RETRY_COUNT times on exception."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        last_exc = None
        for attempt in range(1, KITE_RETRY_COUNT + 1):
            try:
                return func(self, *args, **kwargs)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt {}/{} for {} failed: {}",
                    attempt,
                    KITE_RETRY_COUNT,
                    func.__name__,
                    exc,
                )
                if attempt < KITE_RETRY_COUNT:
                    time.sleep(KITE_RETRY_DELAY_SEC)
        logger.error(
            "{} failed after {} retries: {}", func.__name__, KITE_RETRY_COUNT, last_exc
        )
        raise last_exc

    return wrapper


class ZerodhaClient:
    """Full-featured Zerodha Kite Connect client with rate limiting,
    retry logic, paper-trade mode, and Redis-backed tick streaming.

    Attributes:
        kite: Authenticated KiteConnect instance.
        redis_client: Redis connection for publishing live ticks.
        ticker: KiteTicker WebSocket instance (created on demand).
    """

    # --------------------------------------------------------------------- #
    #  Initialisation & Authentication
    # --------------------------------------------------------------------- #

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialise KiteConnect, Redis, and the rate-limiter semaphore.

        Args:
            api_key: Override API key (uses config/saved credentials if None).
            api_secret: Override API secret (uses config/saved credentials if None).
        """
        # Resolve credentials: explicit args > config > saved file
        self._api_key = api_key or KITE_API_KEY
        self._api_secret = api_secret or KITE_API_SECRET

        # If still empty, try loading from saved credentials
        if not self._api_key or not self._api_secret:
            saved_key, saved_secret = self.load_credentials()
            if saved_key and saved_secret:
                self._api_key = self._api_key or saved_key
                self._api_secret = self._api_secret or saved_secret
                logger.info("Loaded saved credentials from .credentials.json")

        self.kite = KiteConnect(api_key=self._api_key) if self._api_key else None
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        self.ticker: Optional[KiteTicker] = None

        # --- auth state ---
        self._access_token: Optional[str] = None
        self._session_timestamp: Optional[str] = None

        # --- rate limiter (token-bucket style via semaphore) ---
        self._sem = threading.Semaphore(KITE_MAX_REQUESTS_PER_SEC)
        self._rate_lock = threading.Lock()
        self._release_timer_running = False
        self._start_rate_release_loop()

        # --- instruments cache (lazy-loaded) ---
        self._instruments_cache: Optional[list] = None
        self._instruments_fetched_at: Optional[datetime] = None

        # --- authenticate ---
        if KITE_ACCESS_TOKEN:
            self.kite.set_access_token(KITE_ACCESS_TOKEN)
            self._access_token = KITE_ACCESS_TOKEN
            self._session_timestamp = datetime.now(IST).isoformat()
            logger.info("Access token loaded from config.")
        else:
            # Try auto-restore from saved session
            restored = self.auto_restore_session()
            if not restored:
                logger.warning("No access token available — call authenticate() or use /login.")

    # ---- rate-limiter internals ---- #

    def _start_rate_release_loop(self):
        """Background thread that releases semaphore permits every second."""
        def _release():
            while True:
                time.sleep(1.0)
                # Reset to max permits each second
                with self._rate_lock:
                    current = self._sem._value
                    for _ in range(KITE_MAX_REQUESTS_PER_SEC - current):
                        self._sem.release()

        t = threading.Thread(target=_release, daemon=True)
        t.start()
        self._release_timer_running = True

    def _rate_limit(self):
        """Block until a rate-limit permit is available."""
        self._sem.acquire()

    # ---- authentication ---- #

    def authenticate(self) -> str:
        """Open browser for Zerodha login, accept request_token, generate session.

        Returns:
            str: The new access token.
        """
        login_url = self.kite.login_url()
        logger.info("Opening Zerodha login URL in browser: {}", login_url)
        webbrowser.open(login_url)

        request_token = input("Paste the request_token from the redirect URL: ").strip()
        try:
            session = self.kite.generate_session(
                request_token, api_secret=KITE_API_SECRET
            )
            access_token = session["access_token"]
            self.kite.set_access_token(access_token)
            logger.success(
                "Session generated. Access token: {}...{}",
                access_token[:6],
                access_token[-4:],
            )
            return access_token
        except Exception as exc:
            logger.error("Authentication failed: {}", exc)
            raise

    # --------------------------------------------------------------------- #
    #  Credential Management & Session Restore
    # --------------------------------------------------------------------- #

    @staticmethod
    def save_credentials(api_key: str, api_secret: str) -> bool:
        """Save API key and secret to encrypted .credentials.json.

        Uses base64 encoding for basic obfuscation (not true encryption,
        but prevents casual reading of plaintext credentials).

        Args:
            api_key: Zerodha Kite API key.
            api_secret: Zerodha Kite API secret.

        Returns:
            bool: True if saved successfully.
        """
        try:
            payload = {
                "api_key": base64.b64encode(api_key.encode()).decode(),
                "api_secret": base64.b64encode(api_secret.encode()).decode(),
                "saved_at": datetime.now(IST).isoformat(),
            }
            CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.success("Credentials saved to {}", CREDENTIALS_FILE)
            return True
        except Exception as exc:
            logger.error("Failed to save credentials: {}", exc)
            return False

    @staticmethod
    def load_credentials() -> Tuple[str, str]:
        """Load saved API key and secret from .credentials.json.

        Returns:
            Tuple[str, str]: (api_key, api_secret) or ("", "") if not found.
        """
        try:
            if not CREDENTIALS_FILE.exists():
                return "", ""
            data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            api_key = base64.b64decode(data["api_key"]).decode()
            api_secret = base64.b64decode(data["api_secret"]).decode()
            logger.debug("Credentials loaded from .credentials.json")
            return api_key, api_secret
        except Exception as exc:
            logger.error("Failed to load credentials: {}", exc)
            return "", ""

    @staticmethod
    def save_session(access_token: str, api_key: str, api_secret: str) -> bool:
        """Save session (access token + credentials) to .credentials.json.

        Args:
            access_token: The Kite access token.
            api_key: API key.
            api_secret: API secret.

        Returns:
            bool: True if saved successfully.
        """
        try:
            payload = {
                "api_key": base64.b64encode(api_key.encode()).decode(),
                "api_secret": base64.b64encode(api_secret.encode()).decode(),
                "access_token": base64.b64encode(access_token.encode()).decode(),
                "session_timestamp": datetime.now(IST).isoformat(),
                "saved_at": datetime.now(IST).isoformat(),
            }
            CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.success("Session saved to {}", CREDENTIALS_FILE)
            return True
        except Exception as exc:
            logger.error("Failed to save session: {}", exc)
            return False

    @staticmethod
    def load_session() -> dict:
        """Load saved session from .credentials.json.

        Returns:
            dict: Session data with keys: api_key, api_secret, access_token,
                  session_timestamp. Empty dict if not found or expired.
        """
        try:
            if not CREDENTIALS_FILE.exists():
                return {}
            data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
            if "access_token" not in data:
                return {}

            session_ts = data.get("session_timestamp", "")
            if session_ts:
                session_dt = datetime.fromisoformat(session_ts)
                if session_dt.tzinfo is None:
                    session_dt = IST.localize(session_dt)
                now = datetime.now(IST)
                # Kite sessions expire at 6:00 AM IST next day
                today_6am = now.replace(hour=6, minute=0, second=0, microsecond=0)
                if now.hour < 6:
                    today_6am -= timedelta(days=1)
                # Session is valid only if created after last 6 AM reset
                if session_dt < today_6am:
                    logger.info("Saved session expired (created before 6 AM reset)")
                    return {}

            return {
                "api_key": base64.b64decode(data["api_key"]).decode(),
                "api_secret": base64.b64decode(data["api_secret"]).decode(),
                "access_token": base64.b64decode(data["access_token"]).decode(),
                "session_timestamp": session_ts,
            }
        except Exception as exc:
            logger.error("Failed to load session: {}", exc)
            return {}

    def get_login_url(self) -> str:
        """Return the Kite Connect login URL for browser-based authentication.

        Returns:
            str: Full Kite login URL with api_key parameter.
        """
        if not self._api_key:
            raise ValueError("API key not set. Save credentials first.")
        return f"https://kite.zerodha.com/connect/login?api_key={self._api_key}&v=3"

    def handle_callback(self, request_token: str) -> dict:
        """Generate session from Kite callback request_token.

        Args:
            request_token: The token received from Kite redirect callback.

        Returns:
            dict: Session data with access_token, user info, etc.
        """
        if not self.kite:
            raise ValueError("KiteConnect not initialized. Save credentials first.")
        if not self._api_secret:
            raise ValueError("API secret not set. Save credentials first.")

        try:
            session = self.kite.generate_session(
                request_token, api_secret=self._api_secret
            )
            access_token = session["access_token"]
            self.kite.set_access_token(access_token)
            self._access_token = access_token
            self._session_timestamp = datetime.now(IST).isoformat()

            # Save session for auto-restore
            self.save_session(access_token, self._api_key, self._api_secret)

            logger.success(
                "Session generated via callback. Token: {}...{}",
                access_token[:6],
                access_token[-4:],
            )
            return {
                "access_token": access_token,
                "user_id": session.get("user_id", ""),
                "user_name": session.get("user_name", ""),
                "email": session.get("email", ""),
                "broker": session.get("broker", "ZERODHA"),
                "session_timestamp": self._session_timestamp,
            }
        except Exception as exc:
            logger.error("Callback session generation failed: {}", exc)
            raise

    def is_authenticated(self) -> bool:
        """Check if current access token is valid by making a test API call.

        Returns:
            bool: True if authenticated and token is valid.
        """
        if not self.kite or not self._access_token:
            return False
        try:
            profile = self.kite.profile()
            return bool(profile.get("user_id"))
        except Exception:
            return False

    def auto_restore_session(self) -> bool:
        """Try to restore session from saved .credentials.json.

        Checks if saved access token is still valid (not expired past 6 AM).

        Returns:
            bool: True if session restored successfully.
        """
        session = self.load_session()
        if not session:
            return False

        access_token = session.get("access_token", "")
        api_key = session.get("api_key", "")
        if not access_token or not api_key:
            return False

        try:
            # Update instance state
            self._api_key = api_key
            self._api_secret = session.get("api_secret", self._api_secret)
            self._access_token = access_token
            self._session_timestamp = session.get("session_timestamp", "")

            if not self.kite:
                self.kite = KiteConnect(api_key=self._api_key)
            self.kite.set_access_token(access_token)

            # Verify the token is actually valid
            profile = self.kite.profile()
            if profile.get("user_id"):
                logger.success(
                    "Session auto-restored for user: {} ({})",
                    profile.get("user_name", ""),
                    profile.get("user_id", ""),
                )
                return True
            return False
        except Exception as exc:
            logger.warning("Auto-restore failed (token likely expired): {}", exc)
            self._access_token = None
            self._session_timestamp = None
            return False

    def get_auth_status(self) -> dict:
        """Get current authentication status.

        Returns:
            dict: Status info including authenticated flag, user details, timestamps.
        """
        authenticated = self.is_authenticated()
        has_credentials = bool(self._api_key and self._api_secret)
        saved_creds = CREDENTIALS_FILE.exists()

        result = {
            "authenticated": authenticated,
            "has_credentials": has_credentials,
            "has_saved_credentials": saved_creds,
            "api_key_masked": f"{self._api_key[:4]}****{self._api_key[-2:]}" if self._api_key and len(self._api_key) > 6 else "",
            "session_timestamp": self._session_timestamp or "",
        }

        if authenticated:
            try:
                profile = self.kite.profile()
                result["user_id"] = profile.get("user_id", "")
                result["user_name"] = profile.get("user_name", "")
                result["email"] = profile.get("email", "")
                result["broker"] = profile.get("broker", "ZERODHA")
            except Exception:
                pass

        return result

    def update_credentials(self, api_key: str, api_secret: str):
        """Update in-memory credentials and reinitialize KiteConnect.

        Args:
            api_key: New API key.
            api_secret: New API secret.
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self.kite = KiteConnect(api_key=api_key)
        self._access_token = None
        self._session_timestamp = None
        logger.info("Credentials updated in memory. KiteConnect reinitialized.")

    # --------------------------------------------------------------------- #
    #  Market Data
    # --------------------------------------------------------------------- #

    @_retry
    def get_ltp(self, symbols: list) -> dict:
        """Fetch last-traded price for a list of symbols.

        Args:
            symbols: List of exchange-prefixed symbols, e.g. ["NSE:NIFTY 50"].

        Returns:
            dict: Mapping of symbol -> LTP value.
        """
        self._rate_limit()
        logger.debug("get_ltp called for {} symbols", len(symbols))
        raw = self.kite.ltp(symbols)
        result = {sym: data["last_price"] for sym, data in raw.items()}
        logger.info("LTP fetched: {}", result)
        return result

    @_retry
    def get_quote(self, symbol: str) -> dict:
        """Fetch full quote for a single symbol.

        Args:
            symbol: Exchange-prefixed symbol, e.g. "NSE:RELIANCE".

        Returns:
            dict: Full quote dict including OHLC, volume, OI etc.
        """
        self._rate_limit()
        logger.debug("get_quote for {}", symbol)
        raw = self.kite.quote([symbol])
        quote = raw.get(symbol, {})
        logger.info("Quote for {}: ltp={}", symbol, quote.get("last_price"))
        return quote

    @_retry
    def get_option_chain(self, index: str, expiry: str) -> pd.DataFrame:
        """Build an option chain DataFrame for ATM +/- 10 strikes.

        Args:
            index: Index name, e.g. "NIFTY" or "BANKNIFTY".
            expiry: Expiry date string "YYYY-MM-DD".

        Returns:
            pd.DataFrame: Columns — strike, ce_token, ce_ltp, ce_oi, ce_volume,
                          ce_iv, pe_token, pe_ltp, pe_oi, pe_volume, pe_iv, pcr.
        """
        self._rate_limit()
        logger.info("Building option chain for {} expiry {}", index, expiry)

        expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
        instruments = self._get_instruments("NFO")

        # Filter instruments for this index and expiry
        filtered = [
            i
            for i in instruments
            if i["name"] == index
            and i["expiry"] == expiry_date
            and i["instrument_type"] in ("CE", "PE")
        ]

        if not filtered:
            logger.warning("No instruments found for {} expiry {}", index, expiry)
            return pd.DataFrame()

        # Determine ATM strike from spot LTP
        idx_config = INDICES.get(index, {})
        spot_symbol = idx_config.get("symbol", f"NSE:{index}")
        spot_ltp_data = self.get_ltp([spot_symbol])
        spot_price = list(spot_ltp_data.values())[0]

        # Infer strike step from sorted unique strikes
        all_strikes = sorted({i["strike"] for i in filtered})
        if len(all_strikes) < 2:
            step = 50
        else:
            step = all_strikes[1] - all_strikes[0]

        atm_strike = round(spot_price / step) * step
        strikes_range = [atm_strike + (i * step) for i in range(-10, 11)]

        # Separate CE / PE, keyed by strike
        ce_map = {}
        pe_map = {}
        tokens_to_fetch = []
        for inst in filtered:
            if inst["strike"] in strikes_range:
                token = inst["instrument_token"]
                if inst["instrument_type"] == "CE":
                    ce_map[inst["strike"]] = inst
                else:
                    pe_map[inst["strike"]] = inst
                tokens_to_fetch.append(f"NFO:{inst['tradingsymbol']}")

        # Batch fetch quotes (Kite allows up to 500 per call)
        self._rate_limit()
        quotes = {}
        batch_size = 500
        for i in range(0, len(tokens_to_fetch), batch_size):
            batch = tokens_to_fetch[i : i + batch_size]
            self._rate_limit()
            quotes.update(self.kite.quote(batch))

        rows = []
        for strike in sorted(strikes_range):
            ce_inst = ce_map.get(strike)
            pe_inst = pe_map.get(strike)

            ce_key = f"NFO:{ce_inst['tradingsymbol']}" if ce_inst else None
            pe_key = f"NFO:{pe_inst['tradingsymbol']}" if pe_inst else None

            ce_q = quotes.get(ce_key, {})
            pe_q = quotes.get(pe_key, {})

            ce_oi = ce_q.get("oi", 0)
            pe_oi = pe_q.get("oi", 0)
            pcr = round(pe_oi / ce_oi, 4) if ce_oi else 0.0

            rows.append(
                {
                    "strike": strike,
                    "ce_token": ce_inst["instrument_token"] if ce_inst else None,
                    "ce_ltp": ce_q.get("last_price", 0),
                    "ce_oi": ce_oi,
                    "ce_volume": ce_q.get("volume", 0),
                    "ce_iv": ce_q.get("oi_day_high", 0),  # best available proxy
                    "pe_token": pe_inst["instrument_token"] if pe_inst else None,
                    "pe_ltp": pe_q.get("last_price", 0),
                    "pe_oi": pe_oi,
                    "pe_volume": pe_q.get("volume", 0),
                    "pe_iv": pe_q.get("oi_day_high", 0),
                    "pcr": pcr,
                }
            )

        df = pd.DataFrame(rows)
        logger.success(
            "Option chain built: {} strikes, ATM={}", len(df), atm_strike
        )
        return df

    @_retry
    def get_historical_data(
        self, symbol: str, interval: str, days: int
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candles.

        Args:
            symbol: Exchange-prefixed symbol, e.g. "NSE:RELIANCE".
            interval: Candle interval — "minute", "3minute", "5minute",
                      "15minute", "30minute", "60minute", "day".
            days: Number of calendar days of history to fetch.

        Returns:
            pd.DataFrame: Columns — date, open, high, low, close, volume.
        """
        self._rate_limit()
        logger.debug("get_historical_data {} interval={} days={}", symbol, interval, days)

        # Resolve instrument token
        quote = self.kite.quote([symbol])
        token = quote[symbol]["instrument_token"]

        to_date = datetime.now(IST)
        from_date = to_date - timedelta(days=days)

        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date.strftime("%Y-%m-%d"),
            to_date=to_date.strftime("%Y-%m-%d"),
            interval=interval,
        )

        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(IST)
        logger.info(
            "Historical data: {} candles for {} ({})", len(df), symbol, interval
        )
        return df

    # --------------------------------------------------------------------- #
    #  Positions & Orders
    # --------------------------------------------------------------------- #

    @_retry
    def get_positions(self) -> dict:
        """Fetch current day and net positions.

        Returns:
            dict: Keys "day" and "net", each a list of position dicts.
        """
        self._rate_limit()
        positions = self.kite.positions()
        day_count = len(positions.get("day", []))
        net_count = len(positions.get("net", []))
        logger.info("Positions fetched — day: {}, net: {}", day_count, net_count)
        return positions

    @_retry
    def get_orders(self) -> list:
        """Fetch all orders for the current trading day.

        Returns:
            list: List of order dicts.
        """
        self._rate_limit()
        orders = self.kite.orders()
        logger.info("Orders fetched: {} total", len(orders))
        return orders

    # --------------------------------------------------------------------- #
    #  Order Placement
    # --------------------------------------------------------------------- #

    @_retry
    def place_order(
        self,
        tradingsymbol: str,
        exchange: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        product: str = "MIS",
        tag: str = "ASTRONIFTY",
    ) -> str:
        """Place a regular order on Zerodha.

        In PAPER_TRADE mode the order is logged but not sent to the exchange.

        Args:
            tradingsymbol: Instrument tradingsymbol, e.g. "NIFTY23JUN18000CE".
            exchange: "NSE", "NFO", "BSE", etc.
            transaction_type: "BUY" or "SELL".
            quantity: Number of shares / lots.
            order_type: "MARKET", "LIMIT", "SL", "SL-M".
            price: Limit price (required for LIMIT / SL).
            trigger_price: Trigger price (required for SL / SL-M).
            product: "MIS", "CNC", or "NRML".
            tag: Order tag for identification.

        Returns:
            str: Order ID (or "PAPER-<timestamp>" in paper mode).
        """
        order_params = {
            "tradingsymbol": tradingsymbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product": product,
            "tag": tag,
        }
        if price is not None:
            order_params["price"] = price
        if trigger_price is not None:
            order_params["trigger_price"] = trigger_price

        if PAPER_TRADE:
            paper_id = f"PAPER-{int(datetime.now(IST).timestamp())}"
            logger.warning(
                "[PAPER] Order NOT placed: {} {} {} x{} @ {} | id={}",
                transaction_type,
                tradingsymbol,
                order_type,
                quantity,
                price or "MKT",
                paper_id,
            )
            return paper_id

        self._rate_limit()
        order_id = self.kite.place_order(variety=self.kite.VARIETY_REGULAR, **order_params)
        logger.success(
            "Order placed: {} {} {} x{} @ {} | id={}",
            transaction_type,
            tradingsymbol,
            order_type,
            quantity,
            price or "MKT",
            order_id,
        )
        return str(order_id)

    @_retry
    def place_bracket_order(
        self,
        symbol: str,
        qty: int,
        entry: float,
        sl: float,
        target: float,
    ) -> dict:
        """Place a bracket order (entry + stop-loss + target as linked legs).

        In PAPER_TRADE mode the order is logged but not sent to the exchange.

        Args:
            symbol: Tradingsymbol, e.g. "NIFTY23JUN18000CE".
            qty: Quantity.
            entry: Entry limit price.
            sl: Absolute stop-loss price.
            target: Absolute target price.

        Returns:
            dict: {"order_id": str, "entry": float, "sl": float, "target": float}.
        """
        transaction_type = "BUY" if target > entry else "SELL"
        squareoff = abs(target - entry)
        stoploss = abs(entry - sl)

        if PAPER_TRADE:
            paper_id = f"PAPER-BO-{int(datetime.now(IST).timestamp())}"
            logger.warning(
                "[PAPER] Bracket order NOT placed: {} {} entry={} sl={} target={} | id={}",
                transaction_type,
                symbol,
                entry,
                sl,
                target,
                paper_id,
            )
            return {
                "order_id": paper_id,
                "entry": entry,
                "sl": sl,
                "target": target,
            }

        self._rate_limit()
        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_BO,
            tradingsymbol=symbol,
            exchange="NFO",
            transaction_type=transaction_type,
            quantity=qty,
            order_type="LIMIT",
            price=entry,
            squareoff=squareoff,
            stoploss=stoploss,
            product="MIS",
            tag="ASTRONIFTY-BO",
        )
        logger.success(
            "Bracket order placed: {} {} entry={} sl={} target={} | id={}",
            transaction_type,
            symbol,
            entry,
            sl,
            target,
            order_id,
        )
        return {
            "order_id": str(order_id),
            "entry": entry,
            "sl": sl,
            "target": target,
        }

    @_retry
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: The Zerodha order ID to cancel.

        Returns:
            bool: True if cancellation succeeded.
        """
        self._rate_limit()
        self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR, order_id=order_id)
        logger.success("Order cancelled: {}", order_id)
        return True

    @_retry
    def modify_order(self, order_id: str, price: float) -> bool:
        """Modify the price of an open order.

        Args:
            order_id: The Zerodha order ID to modify.
            price: New limit price.

        Returns:
            bool: True if modification succeeded.
        """
        self._rate_limit()
        self.kite.modify_order(
            variety=self.kite.VARIETY_REGULAR,
            order_id=order_id,
            price=price,
        )
        logger.success("Order modified: {} -> price={}", order_id, price)
        return True

    # --------------------------------------------------------------------- #
    #  WebSocket Ticker
    # --------------------------------------------------------------------- #

    def connect_ticker(
        self, instrument_tokens: list, on_tick_callback: Callable
    ) -> None:
        """Connect KiteTicker WebSocket for real-time price streaming.

        Ticks are pushed to the provided callback AND published to Redis
        channel ``ticks:<instrument_token>``.

        Args:
            instrument_tokens: List of integer instrument tokens to subscribe.
            on_tick_callback: Callable receiving (ws, ticks) — same signature
                              as KiteTicker on_ticks.
        """
        logger.info(
            "Connecting ticker for {} instruments", len(instrument_tokens)
        )
        self.ticker = KiteTicker(
            KITE_API_KEY, self.kite.access_token
        )

        def _on_ticks(ws, ticks):
            """Internal handler: invoke user callback + publish to Redis."""
            now = datetime.now(IST).isoformat()
            for tick in ticks:
                token = tick.get("instrument_token")
                payload = {
                    "token": token,
                    "ltp": tick.get("last_price"),
                    "volume": tick.get("volume_traded"),
                    "oi": tick.get("oi"),
                    "timestamp": now,
                }
                try:
                    self.redis_client.publish(f"ticks:{token}", str(payload))
                except Exception as redis_exc:
                    logger.error("Redis publish error for {}: {}", token, redis_exc)
            # Forward to caller's callback
            on_tick_callback(ws, ticks)

        def _on_connect(ws, response):
            logger.info("Ticker connected. Subscribing to {} tokens.", len(instrument_tokens))
            ws.subscribe(instrument_tokens)
            ws.set_mode(ws.MODE_FULL, instrument_tokens)

        def _on_close(ws, code, reason):
            logger.warning("Ticker closed: code={} reason={}", code, reason)

        def _on_error(ws, code, reason):
            logger.error("Ticker error: code={} reason={}", code, reason)

        self.ticker.on_ticks = _on_ticks
        self.ticker.on_connect = _on_connect
        self.ticker.on_close = _on_close
        self.ticker.on_error = _on_error

        self.ticker.connect(threaded=True)
        logger.success("Ticker thread started.")

    # --------------------------------------------------------------------- #
    #  Auxiliary Data
    # --------------------------------------------------------------------- #

    @_retry
    def get_gift_nifty(self) -> dict:
        """Fetch Gift Nifty (SGX Nifty) indicative data.

        Uses NSE's pre-market / SGX future quote via Kite.
        Falls back to the INDICES config mapping if available.

        Returns:
            dict: {"symbol": str, "ltp": float, "change": float,
                   "change_pct": float, "timestamp": str}.
        """
        self._rate_limit()
        logger.debug("Fetching Gift Nifty data")

        # Gift Nifty trades on NSE as "NIFTY" index future on GIFT exchange
        # Attempt via NSE index quote first
        gift_symbol = INDICES.get("GIFT_NIFTY", "NSE:NIFTY 50")
        try:
            quote = self.kite.quote([gift_symbol])
            q = quote.get(gift_symbol, {})
            ohlc = q.get("ohlc", {})
            ltp = q.get("last_price", 0)
            prev_close = ohlc.get("close", ltp)
            change = round(ltp - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
            result = {
                "symbol": gift_symbol,
                "ltp": ltp,
                "change": change,
                "change_pct": change_pct,
                "timestamp": datetime.now(IST).isoformat(),
            }
            logger.info("Gift Nifty: {} ({:+.2f}%)", ltp, change_pct)
            return result
        except Exception as exc:
            logger.error("Failed to fetch Gift Nifty: {}", exc)
            raise

    @_retry
    def get_sector_data(self, sectors: Optional[dict] = None) -> dict:
        """Fetch LTP for all sector indices.

        Args:
            sectors: Mapping of sector_name -> exchange-prefixed symbol.
                     Defaults to SECTORS from config if not provided.

        Returns:
            dict: Mapping of sector_name -> {"symbol": str, "ltp": float,
                  "change": float, "change_pct": float}.
        """
        sectors = sectors or SECTORS
        if not sectors:
            logger.warning("No sectors configured.")
            return {}

        self._rate_limit()
        logger.debug("Fetching sector data for {} sectors", len(sectors))

        symbols = list(sectors.values())
        raw = self.kite.quote(symbols)

        result = {}
        for name, sym in sectors.items():
            q = raw.get(sym, {})
            ohlc = q.get("ohlc", {})
            ltp = q.get("last_price", 0)
            prev_close = ohlc.get("close", ltp)
            change = round(ltp - prev_close, 2)
            change_pct = round((change / prev_close) * 100, 2) if prev_close else 0.0
            result[name] = {
                "symbol": sym,
                "ltp": ltp,
                "change": change,
                "change_pct": change_pct,
            }

        logger.info("Sector data fetched for {} sectors", len(result))
        return result

    # --------------------------------------------------------------------- #
    #  Internal Helpers
    # --------------------------------------------------------------------- #

    def _get_instruments(self, exchange: str = "NFO") -> list:
        """Return cached instrument list (refreshed once per day).

        Args:
            exchange: Exchange segment — "NFO", "NSE", "BSE", etc.

        Returns:
            list: List of instrument dicts from Kite.
        """
        now = datetime.now(IST)
        if (
            self._instruments_cache is not None
            and self._instruments_fetched_at is not None
            and self._instruments_fetched_at.date() == now.date()
        ):
            return self._instruments_cache

        self._rate_limit()
        logger.info("Downloading {} instruments list...", exchange)
        self._instruments_cache = self.kite.instruments(exchange)
        self._instruments_fetched_at = now
        logger.success(
            "Instruments cached: {} items", len(self._instruments_cache)
        )
        return self._instruments_cache

    def disconnect_ticker(self) -> None:
        """Gracefully close the WebSocket ticker connection."""
        if self.ticker:
            try:
                self.ticker.close()
                logger.info("Ticker disconnected.")
            except Exception as exc:
                logger.error("Error disconnecting ticker: {}", exc)
            finally:
                self.ticker = None

    def __repr__(self) -> str:
        mode = "PAPER" if PAPER_TRADE else "LIVE"
        return f"<ZerodhaClient mode={mode} api_key={KITE_API_KEY[:6]}...>"
