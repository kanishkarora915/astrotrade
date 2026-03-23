"""
AstroNifty Engine Scheduler — APScheduler-based job orchestration (Multi-User).

Manages every recurring task the trading engine needs:
    - 60-second market cycle (9:15-15:30 IST, weekdays)
    - Pre-market routine (8:45 AM)
    - Post-market routine (4:00 PM)
    - Session invalidation (6:05 AM — all users)
    - Nightly ML retrain (11:00 PM)
    - Weekly forecast (Sunday 6:00 PM)
    - Trade monitor heartbeat (every 5 seconds — loops ALL authenticated users)
"""

from __future__ import annotations

from datetime import datetime, time
from typing import TYPE_CHECKING

import pytz
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

if TYPE_CHECKING:
    from core.engine import MasterEngine
    from core.user_manager import UserManager

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


class EngineScheduler:
    """Wraps APScheduler ``BackgroundScheduler`` and wires every timed job
    back to the owning :class:`MasterEngine`.

    Multi-user aware: trade monitoring loops all authenticated users,
    session invalidation runs at 6:05 AM for all users.
    """

    def __init__(
        self,
        engine: "MasterEngine",
        user_manager: "UserManager",
    ) -> None:
        self.engine = engine
        self.user_manager = user_manager
        self.scheduler = BackgroundScheduler(timezone=IST)
        self.scheduler.add_listener(self._on_job_event, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
        self._jobs_configured = False
        logger.info("EngineScheduler created (IST timezone, multi-user)")

    # ------------------------------------------------------------------
    # Job definitions
    # ------------------------------------------------------------------
    def setup_jobs(self) -> None:
        """Register every recurring job on the scheduler."""
        if self._jobs_configured:
            logger.warning("setup_jobs() called twice — skipping")
            return

        # ── 1. Main market cycle — every 60 s, weekdays 9:15-15:30 IST ──
        self.scheduler.add_job(
            self._guarded_run_cycle,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour="9-15",
                minute="*",
                second="0",
                timezone=IST,
            ),
            id="market_cycle_60s",
            name="60s Market Cycle",
            max_instances=1,
            misfire_grace_time=30,
        )

        # ── 2. Pre-market routine — 8:45 AM IST, weekdays ───────────────
        self.scheduler.add_job(
            self._guarded_pre_market,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=8,
                minute=45,
                timezone=IST,
            ),
            id="pre_market",
            name="Pre-Market Routine",
            max_instances=1,
            misfire_grace_time=120,
        )

        # ── 3. Post-market routine — 4:00 PM IST, weekdays ──────────────
        self.scheduler.add_job(
            self._guarded_post_market,
            trigger=CronTrigger(
                day_of_week="mon-fri",
                hour=16,
                minute=0,
                timezone=IST,
            ),
            id="post_market",
            name="Post-Market Routine",
            max_instances=1,
            misfire_grace_time=300,
        )

        # ── 4. Session invalidation — 6:05 AM IST, every day ────────────
        #    Kite tokens expire daily; invalidate all user sessions so they
        #    must re-authenticate before the next trading day.
        self.scheduler.add_job(
            self._guarded_session_invalidation,
            trigger=CronTrigger(
                hour=6,
                minute=5,
                timezone=IST,
            ),
            id="session_invalidation",
            name="Invalidate All User Sessions",
            max_instances=1,
            misfire_grace_time=300,
        )

        # ── 5. Nightly ML retrain — 11:00 PM IST, every day ─────────────
        self.scheduler.add_job(
            self._guarded_nightly_ml,
            trigger=CronTrigger(
                hour=23,
                minute=0,
                timezone=IST,
            ),
            id="nightly_ml",
            name="Nightly ML Retrain",
            max_instances=1,
            misfire_grace_time=600,
        )

        # ── 6. Weekly forecast — Sunday 6:00 PM IST ─────────────────────
        self.scheduler.add_job(
            self._guarded_weekly_forecast,
            trigger=CronTrigger(
                day_of_week="sun",
                hour=18,
                minute=0,
                timezone=IST,
            ),
            id="weekly_forecast",
            name="Weekly Forecast",
            max_instances=1,
            misfire_grace_time=600,
        )

        # ── 7. Trade monitor — every 5 s during market hours ────────────
        #    Loops ALL authenticated users and checks their positions.
        self.scheduler.add_job(
            self._guarded_trade_monitor,
            trigger=IntervalTrigger(seconds=5, timezone=IST),
            id="trade_monitor_5s",
            name="Trade Monitor (5s, all users)",
            max_instances=1,
            misfire_grace_time=5,
        )

        self._jobs_configured = True
        logger.info(
            "EngineScheduler: {} jobs configured",
            len(self.scheduler.get_jobs()),
        )

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background scheduler. Calls ``setup_jobs`` if not done."""
        if not self._jobs_configured:
            self.setup_jobs()
        self.scheduler.start()
        logger.info("EngineScheduler STARTED — {} active jobs", len(self.scheduler.get_jobs()))

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("EngineScheduler STOPPED")

    # ------------------------------------------------------------------
    # Market-hours check
    # ------------------------------------------------------------------
    def is_market_hours(self) -> bool:
        """Return True if current IST time falls within 9:15-15:30
        on a weekday (Mon-Fri)."""
        now_ist = datetime.now(IST)
        if now_ist.weekday() >= 5:
            return False
        current_time = now_ist.time()
        return MARKET_OPEN <= current_time <= MARKET_CLOSE

    # ------------------------------------------------------------------
    # Guarded wrappers (run only when appropriate)
    # ------------------------------------------------------------------
    def _guarded_run_cycle(self) -> None:
        """Execute the 60-second cycle only during market hours."""
        if not self.is_market_hours():
            return
        try:
            self.engine.run_cycle()
        except Exception:
            logger.exception("run_cycle FAILED")

    def _guarded_pre_market(self) -> None:
        try:
            self.engine.pre_market_routine()
        except Exception:
            logger.exception("pre_market_routine FAILED")

    def _guarded_post_market(self) -> None:
        try:
            self.engine.post_market_routine()
        except Exception:
            logger.exception("post_market_routine FAILED")

    def _guarded_session_invalidation(self) -> None:
        """6:05 AM — invalidate all user sessions.

        Kite access tokens expire daily. This ensures every user must
        re-authenticate before trading begins. Also stops the RealtimeHub
        ticker since no valid tokens exist until users log back in.
        """
        try:
            logger.info("=" * 40)
            logger.info("SESSION INVALIDATION — 6:05 AM")
            logger.info("=" * 40)

            # Stop the ticker (no valid token)
            try:
                self.engine.realtime_hub.stop()
                logger.info("RealtimeHub stopped for session invalidation")
            except Exception:
                logger.exception("RealtimeHub stop failed during invalidation")

            # Invalidate all sessions
            count = self.user_manager.invalidate_all_sessions()
            logger.info(
                "All user sessions invalidated | count={}",
                count,
            )

        except Exception:
            logger.exception("Session invalidation FAILED")

    def _guarded_nightly_ml(self) -> None:
        try:
            self.engine.trainer.retrain_all()
            logger.info("Nightly ML retrain complete")
        except Exception:
            logger.exception("Nightly ML retrain FAILED")

    def _guarded_weekly_forecast(self) -> None:
        try:
            self.engine.probability_engine.compute_weekly_forecast()
            logger.info("Weekly forecast computed")
        except Exception:
            logger.exception("Weekly forecast FAILED")

    def _guarded_trade_monitor(self) -> None:
        """Run the trade monitor for ALL authenticated users during market hours."""
        if not self.is_market_hours():
            return

        sessions = self.user_manager.get_all_authenticated_sessions()
        if not sessions:
            return

        for session in sessions:
            try:
                # Build current prices from shared RealtimeHub
                current_prices = {}
                open_trades = session.db_manager.get_open_trades(session.user_id)
                if not open_trades:
                    continue

                for trade in open_trades:
                    symbol = trade.get("symbol", "")
                    if symbol:
                        price = self.engine.realtime_hub.get_live_price(symbol)
                        if price > 0:
                            current_prices[symbol] = price

                if current_prices:
                    actions = session.trade_monitor.check_all_positions(current_prices)
                    if actions:
                        logger.info(
                            "[MONITOR][USER:{}] {} actions taken",
                            session.user_id,
                            len(actions),
                        )
            except Exception:
                logger.exception(
                    "Trade monitor tick FAILED for user {}",
                    session.user_id,
                )

    # ------------------------------------------------------------------
    # APScheduler listener
    # ------------------------------------------------------------------
    @staticmethod
    def _on_job_event(event) -> None:
        if event.exception:
            logger.error(
                "Job '{}' raised an exception: {}",
                event.job_id,
                event.exception,
            )
        else:
            logger.trace("Job '{}' executed OK", event.job_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def list_jobs(self) -> list[dict]:
        """Return a summary of all registered jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "trigger": str(job.trigger),
                    "next_run": str(job.next_run_time) if job.next_run_time else None,
                }
            )
        return jobs

    def __repr__(self) -> str:
        running = self.scheduler.running if hasattr(self.scheduler, "running") else False
        return (
            f"<EngineScheduler running={running} "
            f"jobs={len(self.scheduler.get_jobs())} "
            f"market_open={self.is_market_hours()}>"
        )
