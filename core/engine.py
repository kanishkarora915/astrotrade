"""
AstroNifty MasterEngine — THE BRAIN (Multi-User Edition).

Orchestrates the entire trading pipeline every 60 seconds:
    Data Collection -> Analysis -> Cross-Index Confirmation ->
    Sector Impact -> Signal Generation -> Risk Check (per user) ->
    Execution (per user) -> Monitor (per user) -> Save -> Broadcast

SHARED layers (run once):
    Phases 1-5: Data, Analysis, Cross-Index, Sector, Signal Generation
PER-USER layers (loop all authenticated sessions):
    Phases 6-10: Risk, Execute, Monitor, Save (user trades), Broadcast (user data)

Graceful startup / shutdown with SIGTERM handling.
"""

from __future__ import annotations

import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pytz
from loguru import logger

# ── Data layer ───────────────────────────────────────────────────────────
from data.zerodha_client import ZerodhaClient
from database.db_manager import DBManager
from data.astro_feed import AstroFeed
from data.fii_scraper import FIIScraper
from data.global_feed import GlobalFeed

# ── Analysis layer ───────────────────────────────────────────────────────
from analysis.astro_engine import AstroEngine
from analysis.oi_chain import OIChainAnalyzer
from analysis.greeks import GreeksAnalyzer
from analysis.price_action import PriceActionAnalyzer
from analysis.smart_money import SmartMoneyAnalyzer

# ── Scoring & signals ───────────────────────────────────────────────────
from scoring.score_engine import ScoreEngine
from scoring.signal_generator import SignalGenerator

# ── Execution layer ─────────────────────────────────────────────────────
from scoring.strike_selector import StrikeSelector
from risk.position_sizer import PositionSizer
from risk.checklist import PreTradeChecklist

# ── ML / probability ────────────────────────────────────────────────────
from analysis.probability import WeeklyProbabilityEngine
from ml.regime_classifier import RegimeClassifier
from ml.pattern_matcher import PatternMatcher
from ml.trainer import ModelTrainer

# ── Core ─────────────────────────────────────────────────────────────────
from core.signal_bus import SignalBus
from core.scheduler import EngineScheduler
from core.realtime_hub import RealtimeHub

# ── Config ───────────────────────────────────────────────────────────────
from config import (
    INDICES,
    SIGNAL_THRESHOLDS,
    SECTORS,
    SCORING_WEIGHTS,
    TIMEZONE,
    MARKET_OPEN,
    MARKET_CLOSE,
    ENGINE_INTERVAL_SECONDS,
)

# Derive constants from config
SCORE_THRESHOLD = SIGNAL_THRESHOLDS.get("mild_bull", 70)
MAX_PARALLEL_WORKERS = 6
TELEGRAM_ENABLED = True
DASHBOARD_ENABLED = True

if TYPE_CHECKING:
    from core.user_manager import UserManager

IST = pytz.timezone("Asia/Kolkata")


class MasterEngine:
    """Central orchestrator that wires every sub-system together and
    drives the 60-second trading loop.

    Multi-user architecture:
        - SHARED: data collection, analysis, scoring, signal generation
        - PER-USER: risk check, execution, monitoring, P&L, notifications
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, user_manager: "UserManager") -> None:
        logger.info("=" * 60)
        logger.info("ASTRONIFTY MASTER ENGINE — INITIALISING (MULTI-USER)")
        logger.info("=" * 60)

        # ── User manager (central registry of all user sessions) ──────
        self.user_manager = user_manager

        # ── Shared data layer (market data — same for everyone) ───────
        self.db_manager = DBManager()
        self.astro_feed = AstroFeed()
        self.fii_scraper = FIIScraper()
        self.global_feed = GlobalFeed()

        # ── Shared analysis layer ─────────────────────────────────────
        self.astro_engine = AstroEngine()
        self.oi_analyzer = OIChainAnalyzer()
        self.greeks_analyzer = GreeksAnalyzer()
        self.price_analyzer = PriceActionAnalyzer()
        self.smart_money_analyzer = SmartMoneyAnalyzer()

        # ── Shared scoring & signals ──────────────────────────────────
        self.score_engine = ScoreEngine()
        self.signal_generator = SignalGenerator()

        # ── Shared strike selection / checklist ───────────────────────
        self.strike_selector = StrikeSelector()
        self.position_sizer = PositionSizer()
        self.checklist = PreTradeChecklist()

        # ── Shared ML / probability ───────────────────────────────────
        self.probability_engine = WeeklyProbabilityEngine()
        self.regime_classifier = RegimeClassifier()
        self.pattern_matcher = PatternMatcher(self.db_manager)
        self.trainer = ModelTrainer(self.db_manager, None)

        # ── Core wiring ──────────────────────────────────────────────
        self.signal_bus = SignalBus()
        self.scheduler = EngineScheduler(self, user_manager=self.user_manager)

        # ── Real-time hub (WebSocket ticker data distribution) ───────
        #    ONE connection — shared market data for all users
        self.realtime_hub = RealtimeHub(
            signal_bus=self.signal_bus,
            user_manager=self.user_manager,
            dashboard_ws=None,
        )
        self.realtime_hub.register_tick_callback(self._on_tick_lightweight)

        # ── State ────────────────────────────────────────────────────
        self.prev_chains: Dict[str, Any] = {}
        self.cycle_count: int = 0
        self.last_cycle_ts: Optional[str] = None
        self.running: bool = False

        # ── Wire signal bus subscriptions ────────────────────────────
        self._setup_subscriptions()

        # ── Register SIGTERM / SIGINT for graceful shutdown ──────────
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)

        logger.info("MasterEngine initialised — all modules loaded (multi-user mode)")

    # ------------------------------------------------------------------
    # Signal bus wiring
    # ------------------------------------------------------------------
    def _setup_subscriptions(self) -> None:
        self.signal_bus.subscribe("sl_hit", self._on_sl_hit)
        self.signal_bus.subscribe("target_hit", self._on_target_hit)
        self.signal_bus.subscribe("emergency_exit", self._on_emergency_exit)
        self.signal_bus.subscribe("new_signal", self._on_new_signal)
        self.signal_bus.subscribe("tick_update", self._on_tick_update)

    def _on_sl_hit(self, data: dict) -> None:
        logger.warning("SL HIT — {}", data)
        user_id = data.get("user_id")
        if user_id:
            session = self.user_manager.get_session(user_id)
            if session:
                session.auto_exit.handle_sl(data)

    def _on_target_hit(self, data: dict) -> None:
        logger.info("TARGET HIT — {}", data)
        user_id = data.get("user_id")
        if user_id:
            session = self.user_manager.get_session(user_id)
            if session:
                session.auto_exit.handle_target(data)

    def _on_emergency_exit(self, data: dict) -> None:
        logger.critical("EMERGENCY EXIT triggered — closing ALL positions for ALL users")
        for session in self.user_manager.get_all_authenticated_sessions():
            try:
                session.order_manager.close_all_positions(reason="emergency_exit")
                logger.warning(
                    "Emergency exit complete for user {}",
                    session.user_id,
                )
            except Exception:
                logger.exception(
                    "Emergency exit FAILED for user {}",
                    session.user_id,
                )

    def _on_new_signal(self, data: dict) -> None:
        logger.info("New signal received via bus: {}", data.get("symbol"))

    def _on_tick_update(self, data: dict) -> None:
        """Handle tick_update events from RealtimeHub — used for logging."""
        pass

    def _on_tick_lightweight(self, ticks: list) -> None:
        """Lightweight per-tick analysis — SL/target monitoring on every tick.

        Runs on every tick batch (~1 second) from the RealtimeHub.
        Loops through ALL authenticated users and checks THEIR open positions.
        """
        if not self.running:
            return

        try:
            sessions = self.user_manager.get_all_authenticated_sessions()
            if not sessions:
                return

            for session in sessions:
                try:
                    self._check_user_positions_on_tick(session)
                except Exception:
                    logger.debug(
                        "Tick monitoring error for user {} (non-fatal)",
                        session.user_id,
                    )

        except Exception:
            logger.debug("Lightweight tick analysis error (non-fatal)")

    def _check_user_positions_on_tick(self, session) -> None:
        """Check a single user's open positions against live tick prices."""
        open_positions = session.trade_monitor.get_open_positions()
        if not open_positions:
            return

        for pos in open_positions:
            symbol = pos.get("symbol", "")

            # Get live price from shared RealtimeHub cache
            live_price = self.realtime_hub.get_live_price(symbol)
            if live_price <= 0:
                continue

            entry = pos.get("entry_price", 0)
            sl = pos.get("sl", 0)
            target = pos.get("target", 0)
            direction = pos.get("direction", "BUY")

            # Check SL hit
            if direction == "BUY" and sl > 0 and live_price <= sl:
                self.signal_bus.publish("sl_hit", {
                    "user_id": session.user_id,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "exit_price": live_price,
                    "pnl": round(live_price - entry, 2),
                    "source": "realtime_tick",
                })
            elif direction == "SELL" and sl > 0 and live_price >= sl:
                self.signal_bus.publish("sl_hit", {
                    "user_id": session.user_id,
                    "symbol": symbol,
                    "entry": entry,
                    "sl": sl,
                    "exit_price": live_price,
                    "pnl": round(entry - live_price, 2),
                    "source": "realtime_tick",
                })

            # Check target hit
            if direction == "BUY" and target > 0 and live_price >= target:
                self.signal_bus.publish("target_hit", {
                    "user_id": session.user_id,
                    "symbol": symbol,
                    "entry": entry,
                    "target": target,
                    "exit_price": live_price,
                    "pnl": round(live_price - entry, 2),
                    "source": "realtime_tick",
                })
            elif direction == "SELL" and target > 0 and live_price <= target:
                self.signal_bus.publish("target_hit", {
                    "user_id": session.user_id,
                    "symbol": symbol,
                    "entry": entry,
                    "target": target,
                    "exit_price": live_price,
                    "pnl": round(entry - live_price, 2),
                    "source": "realtime_tick",
                })

            # Publish live P&L update (per user)
            if direction == "BUY":
                pnl = round((live_price - entry) * pos.get("quantity", 1), 2)
            else:
                pnl = round((entry - live_price) * pos.get("quantity", 1), 2)

            self.signal_bus.publish("position_pnl_update", {
                "user_id": session.user_id,
                "symbol": symbol,
                "ltp": live_price,
                "entry": entry,
                "pnl": pnl,
                "direction": direction,
            })

    # ------------------------------------------------------------------
    # MAIN 60-SECOND CYCLE
    # ------------------------------------------------------------------
    def run_cycle(self) -> None:
        """Execute one full analysis-to-execution cycle.

        SHARED (run once): Phases 1-5
        PER-USER (loop all authenticated users): Phases 6-10
        """
        cycle_start = time.time()
        self.cycle_count += 1
        ts = datetime.now(IST).isoformat()
        self.last_cycle_ts = ts
        logger.info("━" * 50)
        logger.info("CYCLE #{} — {}", self.cycle_count, ts)
        logger.info("━" * 50)

        try:
            # ══════════════════════════════════════════════════════════
            #  SHARED PHASES (run ONCE — same data for all users)
            # ══════════════════════════════════════════════════════════

            # ── PHASE 1: DATA COLLECTION (parallel) ──────────────────
            raw = self._phase_data_collection()

            # ── PHASE 2: ANALYSIS (per index) ────────────────────────
            index_scores = self._phase_analysis(raw)

            # ── PHASE 3: CROSS-INDEX CONFIRMATION ────────────────────
            confirmed = self._phase_cross_index(index_scores)

            # ── PHASE 4: SECTOR IMPACT ───────────────────────────────
            sector_bias = self._phase_sector_impact(raw.get("sector_data"))

            # ── PHASE 5: SIGNAL GENERATION ───────────────────────────
            signals = self._phase_signal_generation(confirmed, sector_bias)

            # ══════════════════════════════════════════════════════════
            #  PER-USER PHASES (loop all authenticated users)
            # ══════════════════════════════════════════════════════════

            sessions = self.user_manager.get_all_authenticated_sessions()
            if not sessions:
                logger.info("No authenticated users — skipping execution phases")
            else:
                logger.info(
                    "Processing {} authenticated users for execution",
                    len(sessions),
                )

            for session in sessions:
                user_id = session.user_id
                try:
                    # Store shared scores on the session so dashboard can read them
                    session.scores = index_scores

                    # ── PHASE 6: RISK CHECK (per user) ───────────────
                    approved = self._phase_risk_check_for_user(signals, session)

                    # ── PHASE 7: EXECUTION (per user) ────────────────
                    self._phase_execution_for_user(approved, session)

                    # ── PHASE 8: MONITOR (per user) ──────────────────
                    self._phase_monitor_for_user(session)

                    # Store signals on session for reference
                    for sig in signals:
                        session.signals.append(sig)

                    logger.info(
                        "[USER:{}] Cycle phases 6-8 complete | approved={}/{}",
                        user_id,
                        len(approved),
                        len(signals),
                    )

                except Exception:
                    logger.exception(
                        "[USER:{}] Per-user cycle phases FAILED",
                        user_id,
                    )

            # ══════════════════════════════════════════════════════════
            #  SAVE & BROADCAST (shared snapshot + per-user trades)
            # ══════════════════════════════════════════════════════════

            # ── PHASE 9: SAVE ────────────────────────────────────────
            self._phase_save(raw, index_scores, signals)
            # Save per-user trade snapshots
            for session in sessions:
                try:
                    self._phase_save_user_trades(session)
                except Exception:
                    logger.exception(
                        "[USER:{}] Per-user save FAILED",
                        session.user_id,
                    )

            # ── PHASE 10: BROADCAST ──────────────────────────────────
            self._phase_broadcast(index_scores, signals, sessions)

        except Exception:
            logger.exception("CYCLE #{} FAILED", self.cycle_count)

        elapsed = round(time.time() - cycle_start, 2)
        logger.info("Cycle #{} complete in {}s", self.cycle_count, elapsed)

    # ------------------------------------------------------------------
    # SHARED Phase implementations
    # ------------------------------------------------------------------
    def _phase_data_collection(self) -> Dict[str, Any]:
        """PHASE 1 — Fetch all raw data in parallel (SHARED)."""
        logger.info("[P1] Data collection START")
        results: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as pool:
            futures = {
                pool.submit(self._fetch_option_chains): "chains",
                pool.submit(self.astro_feed.get_snapshot): "astro",
                pool.submit(self.fii_scraper.fetch_latest): "fii_dii",
                pool.submit(self.global_feed.fetch_cues): "global_cues",
                pool.submit(self.global_feed.fetch_sector_data): "sector_data",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result(timeout=30)
                    logger.debug("[P1] {} fetched OK", key)
                except Exception:
                    logger.error("[P1] {} fetch FAILED: {}", key, traceback.format_exc())
                    results[key] = None

        logger.info("[P1] Data collection DONE — {} sources", len(results))
        return results

    def _fetch_option_chains(self) -> Dict[str, Any]:
        """Fetch option chains — uses RealtimeHub live cache if available,
        falls back to API call via any authenticated user's Kite client."""
        chains: Dict[str, Any] = {}

        # Get a Kite client for API fallback (any authenticated user)
        fallback_kite = self.realtime_hub.get_active_kite_client()

        for idx in INDICES:
            # Try real-time cache first (zero API calls)
            live_chain = self.realtime_hub.get_live_chain(idx)
            if not live_chain.empty:
                chain = live_chain
                previous = self.realtime_hub.get_previous_chain(idx)
                logger.debug("[P1] {} chain from REALTIME cache ({} strikes)", idx, len(chain))
            elif fallback_kite:
                # Fallback to API (cold start or hub not connected)
                chain = fallback_kite.get_option_chain(idx)
                previous = self.prev_chains.get(idx)
                logger.debug(
                    "[P1] {} chain from API fallback ({} strikes)",
                    idx,
                    len(chain) if hasattr(chain, "__len__") else 0,
                )
            else:
                logger.warning("[P1] No kite client available for {} chain fetch", idx)
                chain = None
                previous = self.prev_chains.get(idx)

            chains[idx] = {
                "current": chain,
                "previous": previous or self.prev_chains.get(idx),
            }
        # Rotate: current becomes previous for next cycle
        self.prev_chains = {idx: chains[idx]["current"] for idx in INDICES if chains[idx]["current"] is not None}
        return chains

    def _phase_analysis(self, raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """PHASE 2 — Run all analysis modules for each index (SHARED)."""
        logger.info("[P2] Analysis START")
        index_scores: Dict[str, Dict[str, Any]] = {}
        chains = raw.get("chains") or {}
        astro = raw.get("astro")
        fii_dii = raw.get("fii_dii")

        for idx in INDICES:
            chain_data = chains.get(idx)
            if not chain_data or not chain_data.get("current"):
                logger.warning("[P2] No chain data for {} — skipping", idx)
                continue

            current = chain_data["current"]
            previous = chain_data["previous"]

            # Individual module scores
            astro_score = self.astro_engine.score(astro, idx)
            oi_score = self.oi_analyzer.score(current, previous)
            greeks_score = self.greeks_analyzer.score(current)
            price_score = self.price_analyzer.score(current, idx)
            smart_money_score = self.smart_money_analyzer.score(current, fii_dii)

            # ML overlays
            regime = self.regime_classifier.classify(current, idx)
            pattern = self.pattern_matcher.match(current, idx)
            probability = self.probability_engine.compute(current, idx)

            # Composite
            components = {
                "astro": astro_score,
                "oi": oi_score,
                "greeks": greeks_score,
                "price": price_score,
                "smart_money": smart_money_score,
                "regime": regime,
                "pattern": pattern,
                "probability": probability,
            }
            composite = self.score_engine.compute_composite(components)

            index_scores[idx] = {
                "composite": composite,
                "components": components,
                "direction": composite.get("direction"),
                "confidence": composite.get("confidence", 0),
            }

            self.signal_bus.publish("score_update", {"index": idx, **index_scores[idx]})

        logger.info("[P2] Analysis DONE — {} indices scored", len(index_scores))
        return index_scores

    def _phase_cross_index(self, index_scores: Dict[str, Dict]) -> Dict[str, Dict]:
        """PHASE 3 — Keep only indices with 2-of-3 directional agreement (SHARED)."""
        logger.info("[P3] Cross-index confirmation")

        directions = {}
        for idx in ["NIFTY", "BANKNIFTY", "GIFTNIFTY"]:
            if idx in index_scores:
                directions[idx] = index_scores[idx].get("direction")

        if len(directions) < 2:
            logger.warning("[P3] Not enough indices for cross-confirmation — passing all through")
            return index_scores

        bullish_count = sum(1 for d in directions.values() if d == "BULLISH")
        bearish_count = sum(1 for d in directions.values() if d == "BEARISH")

        agreed_direction = None
        if bullish_count >= 2:
            agreed_direction = "BULLISH"
        elif bearish_count >= 2:
            agreed_direction = "BEARISH"

        if agreed_direction is None:
            logger.info("[P3] No 2-of-3 agreement — no trades this cycle")
            return {}

        confirmed = {
            idx: data
            for idx, data in index_scores.items()
            if data.get("direction") == agreed_direction
        }
        logger.info(
            "[P3] Agreed direction: {} — {} indices confirmed",
            agreed_direction,
            len(confirmed),
        )
        return confirmed

    def _phase_sector_impact(self, sector_data: Any) -> Dict[str, Any]:
        """PHASE 4 — Analyze sector breadth (SHARED)."""
        logger.info("[P4] Sector impact analysis")
        if not sector_data:
            return {"bias": "NEUTRAL", "strength": 0}

        try:
            advancing = sector_data.get("advancing_sectors", 0)
            declining = sector_data.get("declining_sectors", 0)
            total = advancing + declining or 1
            breadth = (advancing - declining) / total

            if breadth > 0.3:
                bias = "BULLISH"
            elif breadth < -0.3:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            result = {"bias": bias, "strength": round(abs(breadth), 2), "raw": sector_data}
            self.signal_bus.publish("sector_update", result)
            logger.info("[P4] Sector bias: {} (strength {})", bias, result["strength"])
            return result
        except Exception:
            logger.exception("[P4] Sector analysis failed")
            return {"bias": "NEUTRAL", "strength": 0}

    def _phase_signal_generation(
        self,
        confirmed: Dict[str, Dict],
        sector_bias: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """PHASE 5 — Generate actionable trading signals (SHARED).

        These signals are the same for all users. Per-user risk/execution
        happens in phases 6-7.
        """
        logger.info("[P5] Signal generation (SHARED)")
        signals: List[Dict[str, Any]] = []

        for idx, data in confirmed.items():
            confidence = data.get("confidence", 0)
            if confidence < SCORE_THRESHOLD:
                logger.info("[P5] {} confidence {}% < threshold {}% — skip", idx, confidence, SCORE_THRESHOLD)
                continue

            sig = self.signal_generator.generate(
                index=idx,
                score_data=data,
                sector_bias=sector_bias,
            )
            if sig:
                strike = self.strike_selector.select(
                    index=idx,
                    direction=data["direction"],
                    chain=self.prev_chains.get(idx),
                )
                sig["strike"] = strike
                signals.append(sig)
                self.signal_bus.publish("new_signal", sig)

        logger.info("[P5] Generated {} signals", len(signals))
        return signals

    # ------------------------------------------------------------------
    # PER-USER Phase implementations
    # ------------------------------------------------------------------
    def _phase_risk_check_for_user(
        self,
        signals: List[Dict],
        session,
    ) -> List[Dict]:
        """PHASE 6 — Pre-trade risk validation for a SPECIFIC user."""
        user_id = session.user_id
        logger.info("[P6][USER:{}] Risk check", user_id)
        approved: List[Dict] = []

        for sig in signals:
            # Make a copy so per-user modifications do not affect shared signal
            user_sig = dict(sig)

            # Position sizing using shared sizer
            size = self.position_sizer.compute(user_sig)
            user_sig["quantity"] = size

            # Checklist gate (shared rules)
            checklist_ok, reasons = self.checklist.validate(user_sig)
            if not checklist_ok:
                logger.warning(
                    "[P6][USER:{}] {} REJECTED by checklist: {}",
                    user_id,
                    user_sig.get("symbol"),
                    reasons,
                )
                continue

            # Per-user risk manager gate
            approval = session.risk_manager.pre_trade_check(user_sig, session.capital)
            if not approval.get("approved"):
                logger.warning(
                    "[P6][USER:{}] {} REJECTED by risk: {}",
                    user_id,
                    user_sig.get("symbol"),
                    approval.get("reason"),
                )
                continue

            # Apply risk-adjusted quantity
            if approval.get("adjusted_quantity"):
                user_sig["quantity"] = approval["adjusted_quantity"]

            user_sig["user_id"] = user_id
            approved.append(user_sig)

        logger.info(
            "[P6][USER:{}] {}/{} signals approved",
            user_id,
            len(approved),
            len(signals),
        )
        return approved

    def _phase_execution_for_user(
        self,
        approved: List[Dict],
        session,
    ) -> None:
        """PHASE 7 — Place orders via the user's OWN Kite client."""
        user_id = session.user_id
        logger.info("[P7][USER:{}] Execution — {} orders to place", user_id, len(approved))

        for sig in approved:
            try:
                result = session.order_manager.execute_signal(sig)
                order_id = result.get("trade_id")
                self.signal_bus.publish(
                    "trade_executed",
                    {
                        "user_id": user_id,
                        "symbol": sig.get("symbol"),
                        "order_id": order_id,
                        "signal": sig,
                    },
                )
                logger.info(
                    "[P7][USER:{}] ORDER PLACED — {} | trade_id={}",
                    user_id,
                    sig.get("symbol"),
                    order_id,
                )
            except Exception:
                logger.exception(
                    "[P7][USER:{}] Order FAILED for {}",
                    user_id,
                    sig.get("symbol"),
                )

    def _phase_monitor_for_user(self, session) -> None:
        """PHASE 8 — Check this user's open positions for SL / target hits."""
        user_id = session.user_id
        logger.info("[P8][USER:{}] Position monitor", user_id)
        try:
            # Build current prices from shared RealtimeHub
            current_prices = self._build_prices_for_user(session)
            if not current_prices:
                return

            events = session.trade_monitor.check_all_positions(current_prices)
            for event in events:
                event["user_id"] = user_id
                event_type = event.get("action")
                if event_type == "SL_HIT":
                    self.signal_bus.publish("sl_hit", event)
                elif event_type == "TARGET_HIT":
                    self.signal_bus.publish("target_hit", event)
        except Exception:
            logger.exception("[P8][USER:{}] Monitor check failed", user_id)

    def _build_prices_for_user(self, session) -> Dict[str, float]:
        """Build a {symbol: price} dict from the shared RealtimeHub cache
        for all symbols this user has open positions in."""
        prices = {}
        try:
            open_trades = session.db_manager.get_open_trades(session.user_id)
            if not open_trades:
                return prices
            for trade in open_trades:
                symbol = trade.get("symbol", "")
                if symbol:
                    price = self.realtime_hub.get_live_price(symbol)
                    if price > 0:
                        prices[symbol] = price
        except Exception:
            pass
        return prices

    # ------------------------------------------------------------------
    # SAVE phases
    # ------------------------------------------------------------------
    def _phase_save(
        self,
        raw: Dict[str, Any],
        index_scores: Dict[str, Dict],
        signals: List[Dict],
    ) -> None:
        """PHASE 9a — Persist SHARED cycle snapshot to the database."""
        logger.info("[P9] Saving shared snapshot")
        try:
            snapshot = {
                "cycle": self.cycle_count,
                "ts": self.last_cycle_ts,
                "scores": index_scores,
                "signals": signals,
                "astro": raw.get("astro"),
                "fii_dii": raw.get("fii_dii"),
                "global_cues": raw.get("global_cues"),
            }
            self.db_manager.save_snapshot(snapshot)
        except Exception:
            logger.exception("[P9] Shared snapshot save failed")

    def _phase_save_user_trades(self, session) -> None:
        """PHASE 9b — Persist per-user trade data."""
        try:
            # The per-user order_manager already saves trades to DB
            # with user_id tagging. This is a hook for any additional
            # per-user snapshots (e.g., portfolio state).
            if hasattr(session, "save_portfolio_snapshot"):
                session.save_portfolio_snapshot()
        except Exception:
            logger.exception(
                "[P9][USER:{}] Per-user save failed",
                session.user_id,
            )

    # ------------------------------------------------------------------
    # BROADCAST phase
    # ------------------------------------------------------------------
    def _phase_broadcast(
        self,
        index_scores: Dict[str, Dict],
        signals: List[Dict],
        sessions: list,
    ) -> None:
        """PHASE 10 — Push shared data to all, user-specific data to each user."""
        logger.info("[P10] Broadcasting")

        # ── Shared broadcast (scores, signals) to dashboard ──────────
        shared_payload = {
            "cycle": self.cycle_count,
            "ts": self.last_cycle_ts,
            "scores": index_scores,
            "signals": signals,
        }
        try:
            if DASHBOARD_ENABLED:
                self.db_manager.push_to_dashboard(shared_payload)
        except Exception:
            logger.exception("[P10] Dashboard push failed")

        # ── Per-user broadcast (Telegram, user-specific dashboard) ───
        for session in sessions:
            try:
                if TELEGRAM_ENABLED and signals and session.telegram_notifier:
                    for sig in signals:
                        session.telegram_notifier.send_signal_sync(sig)
            except Exception:
                logger.exception(
                    "[P10][USER:{}] Telegram push failed",
                    session.user_id,
                )

            try:
                # Push user-specific data (positions, P&L) via dashboard WS
                if DASHBOARD_ENABLED and self.realtime_hub.dashboard_ws:
                    user_payload = {
                        "channel": "user_data",
                        "user_id": session.user_id,
                        "scores": index_scores,
                        "signals": signals,
                        "positions": session.get_open_trades_summary(),
                        "pnl": session.get_daily_pnl(),
                    }
                    self.realtime_hub._broadcast_async(user_payload)
            except Exception:
                logger.exception(
                    "[P10][USER:{}] User dashboard push failed",
                    session.user_id,
                )

    # ------------------------------------------------------------------
    # Pre-market / Post-market
    # ------------------------------------------------------------------
    def pre_market_routine(self) -> None:
        """Run at 8:45 AM IST — fetch Gift Nifty, global cues, daily astro.
        This is SHARED data — same for all users."""
        logger.info("=" * 50)
        logger.info("PRE-MARKET ROUTINE")
        logger.info("=" * 50)
        try:
            gift_nifty = self.global_feed.fetch_gift_nifty()
            global_cues = self.global_feed.fetch_cues()
            astro_today = self.astro_feed.get_daily_forecast()
            regime = self.regime_classifier.classify_pre_market(global_cues)

            summary = {
                "gift_nifty": gift_nifty,
                "global_cues": global_cues,
                "astro_today": astro_today,
                "regime": regime,
                "ts": datetime.now(IST).isoformat(),
            }
            self.db_manager.save_pre_market(summary)
            logger.info("Pre-market data saved: Gift Nifty={}", gift_nifty)

            # Send pre-market summary to ALL users who have Telegram enabled
            if TELEGRAM_ENABLED:
                for session in self.user_manager.get_all_authenticated_sessions():
                    try:
                        if session.telegram_notifier:
                            session.telegram_notifier.send_pre_market_sync(summary)
                    except Exception:
                        logger.exception(
                            "[PRE-MARKET] Telegram send failed for user {}",
                            session.user_id,
                        )

        except Exception:
            logger.exception("Pre-market routine FAILED")

    def post_market_routine(self) -> None:
        """Run at 4:00 PM IST — per-user: close positions, export, cleanup."""
        logger.info("=" * 50)
        logger.info("POST-MARKET ROUTINE (MULTI-USER)")
        logger.info("=" * 50)

        sessions = self.user_manager.get_all_authenticated_sessions()

        for session in sessions:
            user_id = session.user_id
            try:
                logger.info("[POST-MARKET][USER:{}] Processing...", user_id)

                # 1. Close any remaining positions for THIS user
                open_positions = session.trade_monitor.get_open_positions()
                if open_positions:
                    logger.warning(
                        "[POST-MARKET][USER:{}] Closing {} remaining positions",
                        user_id,
                        len(open_positions),
                    )
                    session.order_manager.close_all_positions(reason="eod_exit")

                # 2. End-of-day cleanup via auto_exit
                eod_summary = session.auto_exit.end_of_day_cleanup()

                # 3. Generate daily export (PDF, CSV) for this user
                try:
                    session.exporter.export_daily(session.db_manager, user_id=user_id)
                except Exception:
                    logger.exception(
                        "[POST-MARKET][USER:{}] Export failed",
                        user_id,
                    )

                # 4. Send daily summary via Telegram
                if TELEGRAM_ENABLED and session.telegram_notifier:
                    try:
                        session.telegram_notifier.send_daily_summary_sync(eod_summary)
                    except Exception:
                        logger.exception(
                            "[POST-MARKET][USER:{}] Telegram summary failed",
                            user_id,
                        )

                logger.info("[POST-MARKET][USER:{}] Complete", user_id)

            except Exception:
                logger.exception(
                    "[POST-MARKET][USER:{}] FAILED",
                    user_id,
                )

        # Shared cleanup
        try:
            self.db_manager.eod_cleanup()
        except Exception:
            logger.exception("Shared EOD cleanup failed")

        logger.info("Post-market routine complete for {} users", len(sessions))

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Boot the engine: start real-time hub, start scheduler."""
        logger.info("=" * 60)
        logger.info("ASTRONIFTY MASTER ENGINE — STARTING (MULTI-USER)")
        logger.info("=" * 60)

        # Start the real-time tick hub (uses first authenticated user's token)
        try:
            self.realtime_hub.start()
            logger.info("RealtimeHub started — live tick stream active")
        except Exception:
            logger.exception("RealtimeHub start failed — running in polling mode")

        self.running = True
        self.scheduler.start()
        logger.info("Engine is LIVE — multi-user real-time mode enabled")

    def stop(self) -> None:
        """Graceful shutdown: close ALL users' positions, stop scheduler, flush DB."""
        logger.info("=" * 60)
        logger.info("ASTRONIFTY MASTER ENGINE — SHUTTING DOWN")
        logger.info("=" * 60)
        self.running = False

        # Stop real-time hub first
        try:
            self.realtime_hub.stop()
        except Exception:
            logger.exception("RealtimeHub stop failed")

        # Stop scheduler to prevent new jobs
        try:
            self.scheduler.stop()
        except Exception:
            logger.exception("Scheduler stop failed")

        # Close all open positions for ALL users
        for session in self.user_manager.get_all_authenticated_sessions():
            try:
                open_positions = session.trade_monitor.get_open_positions()
                if open_positions:
                    logger.warning(
                        "Closing {} positions on shutdown for user {}",
                        len(open_positions),
                        session.user_id,
                    )
                    session.order_manager.close_all_positions(reason="engine_shutdown")
            except Exception:
                logger.exception(
                    "Failed to close positions on shutdown for user {}",
                    session.user_id,
                )

        # Flush database
        try:
            self.db_manager.flush()
        except Exception:
            logger.exception("DB flush failed on shutdown")

        # Clear signal bus
        self.signal_bus.clear()

        logger.info("Engine shutdown complete")

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """SIGTERM / SIGINT handler — triggers graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.warning("Received {} — initiating graceful shutdown", sig_name)
        self.stop()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        """Return health-check dict for all modules + per-user status."""
        module_checks = {}
        modules = {
            "db_manager": self.db_manager,
            "astro_feed": self.astro_feed,
            "astro_engine": self.astro_engine,
            "oi_analyzer": self.oi_analyzer,
            "greeks_analyzer": self.greeks_analyzer,
            "price_analyzer": self.price_analyzer,
            "smart_money_analyzer": self.smart_money_analyzer,
            "fii_scraper": self.fii_scraper,
            "global_feed": self.global_feed,
            "score_engine": self.score_engine,
            "signal_generator": self.signal_generator,
            "strike_selector": self.strike_selector,
            "position_sizer": self.position_sizer,
            "checklist": self.checklist,
            "probability_engine": self.probability_engine,
            "regime_classifier": self.regime_classifier,
            "pattern_matcher": self.pattern_matcher,
            "trainer": self.trainer,
            "realtime_hub": self.realtime_hub,
        }

        for name, mod in modules.items():
            try:
                if hasattr(mod, "health_check"):
                    module_checks[name] = mod.health_check()
                else:
                    module_checks[name] = {"status": "loaded", "has_health_check": False}
            except Exception as e:
                module_checks[name] = {"status": "error", "error": str(e)}

        # Per-user status
        user_statuses = {}
        for session in self.user_manager.get_all_authenticated_sessions():
            user_statuses[session.user_id] = {
                "authenticated": True,
                "open_trades": len(session.trade_monitor.get_open_positions() or []),
                "daily_pnl": session.get_daily_pnl(),
            }

        return {
            "engine_running": self.running,
            "cycle_count": self.cycle_count,
            "last_cycle_ts": self.last_cycle_ts,
            "scheduler": repr(self.scheduler),
            "signal_bus": self.signal_bus.all_subscriber_counts(),
            "market_hours": self.scheduler.is_market_hours(),
            "modules": module_checks,
            "authenticated_users": len(user_statuses),
            "users": user_statuses,
        }

    def set_dashboard_ws(self, ws_manager) -> None:
        """Wire the dashboard WebSocket manager into the RealtimeHub."""
        self.realtime_hub.dashboard_ws = ws_manager
        logger.info("Dashboard WebSocket wired into RealtimeHub")

    def __repr__(self) -> str:
        return (
            f"<MasterEngine running={self.running} "
            f"cycles={self.cycle_count} "
            f"users={len(self.user_manager.get_all_authenticated_sessions())} "
            f"last={self.last_cycle_ts}>"
        )
