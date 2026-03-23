"""
AstroNifty Risk Manager — Per-user risk gate (Multi-User).
------------------------------------------------------------
Each RiskManager instance is bound to a SPECIFIC user_id.
All DB queries are filtered by user_id so each user has isolated:
    - Daily loss limits
    - Trade count limits
    - Consecutive loss tracking
    - Position duplication checks

Central risk gate that every signal must pass before execution.
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytz
from loguru import logger

try:
    from config import RISK_RULES, PAPER_TRADE
except ImportError:
    RISK_RULES = {
        "min_score": 70,
        "max_daily_trades": 5,
        "daily_loss_limit_pct": 3.0,
        "max_vix": 20,
        "vix_reduce_threshold": 18,
        "cutoff_hour": 15,
        "cutoff_minute": 0,
        "announcement_buffer_mins": 15,
        "consecutive_loss_pause_mins": 90,
        "index_agreement_min": 2,
    }
    PAPER_TRADE = True

IST = pytz.timezone("Asia/Kolkata")

MAJOR_ANNOUNCEMENTS: List[datetime] = [
    IST.localize(datetime(2026, 4, 9, 10, 0)),
    IST.localize(datetime(2026, 6, 6, 10, 0)),
    IST.localize(datetime(2026, 8, 7, 10, 0)),
    IST.localize(datetime(2026, 10, 8, 10, 0)),
    IST.localize(datetime(2026, 12, 5, 10, 0)),
]


class RiskManager:
    """Per-user gate-keeper that approves / blocks every trade signal.

    Each user gets their own RiskManager with their own:
        - capital
        - daily P&L tracking
        - trade count limits
        - lock / pause state
    """

    def __init__(self, db_manager: Any, capital: float, user_id: str) -> None:
        """
        Args:
            db_manager: Database helper (shared, queries filtered by user_id).
            capital:    This user's trading capital.
            user_id:    The user this risk manager serves.
        """
        self.db = db_manager
        self.capital = capital
        self.user_id = user_id
        self._locked = False
        self._pause_until: Optional[datetime] = None
        logger.info(
            "RiskManager initialised | user={} | capital={} | paper={}",
            user_id,
            capital,
            PAPER_TRADE,
        )

    # ------------------------------------------------------------------
    # 1. Pre-trade check (10-point gate)
    # ------------------------------------------------------------------
    def pre_trade_check(self, signal: dict, capital: float) -> dict:
        """
        Run all 10 checks against *signal* for THIS user.

        Parameters
        ----------
        signal : dict
            Must contain at minimum:
                score, direction, index, entry, sl, lot_size,
                margin_required, astro_window_start, astro_window_end,
                vix, indices_bullish, indices_bearish
        capital : float
            Current available capital for THIS user.

        Returns
        -------
        dict with keys:
            approved, checks_passed, checks_failed,
            adjusted_quantity, reason
        """
        try:
            now_ist = datetime.now(IST)
            checks_passed: List[str] = []
            checks_failed: List[str] = []
            adjusted_quantity: int = signal.get("quantity", 1)
            vix: float = signal.get("vix", 0.0)

            # --- Check 0: System locked/paused for this user ---
            if self._locked:
                return {
                    "approved": False,
                    "checks_passed": [],
                    "checks_failed": ["system_locked"],
                    "adjusted_quantity": 0,
                    "reason": f"BLOCKED: System locked for user {self.user_id}",
                }

            if self._pause_until and now_ist < self._pause_until:
                return {
                    "approved": False,
                    "checks_passed": [],
                    "checks_failed": ["system_paused"],
                    "adjusted_quantity": 0,
                    "reason": f"BLOCKED: Paused until {self._pause_until.strftime('%H:%M IST')}",
                }

            # --- Check 1: Score >= 70 ---
            if signal.get("score", 0) >= RISK_RULES.get("min_score", 70):
                checks_passed.append("score_threshold")
            else:
                checks_failed.append("score_threshold")

            # --- Check 2: Time < 15:25 IST ---
            no_entry_str = RISK_RULES.get("no_entry_after", "15:25")
            cutoff_parts = no_entry_str.split(":")
            cutoff = now_ist.replace(
                hour=int(cutoff_parts[0]),
                minute=int(cutoff_parts[1]),
                second=0,
                microsecond=0,
            )
            if now_ist < cutoff:
                checks_passed.append("time_cutoff")
            else:
                checks_failed.append("time_cutoff")

            # --- Check 3: Not within 15 mins of major announcement ---
            buffer = timedelta(minutes=RISK_RULES.get("announcement_buffer_mins", 15))
            near_announcement = False
            for ann in MAJOR_ANNOUNCEMENTS:
                if abs(now_ist - ann) <= buffer:
                    near_announcement = True
                    break
            if not near_announcement:
                checks_passed.append("announcement_buffer")
            else:
                checks_failed.append("announcement_buffer")

            # --- Check 4: Daily P&L > -3% of capital (per user) ---
            daily_pnl = self._get_today_pnl()
            loss_limit = -(RISK_RULES.get("daily_loss_limit_pct", 3.0) / 100.0) * capital
            if daily_pnl > loss_limit:
                checks_passed.append("daily_pnl_limit")
            else:
                checks_failed.append("daily_pnl_limit")

            # --- Check 5: Less than 5 trades today (per user) ---
            trade_count = self._get_today_trade_count()
            if trade_count < RISK_RULES.get("max_daily_trades", 5):
                checks_passed.append("max_daily_trades")
            else:
                checks_failed.append("max_daily_trades")

            # --- Check 6: Margin sufficient (per user capital) ---
            margin_required = signal.get("margin_required", 0.0)
            if capital >= margin_required:
                checks_passed.append("margin_sufficient")
            else:
                checks_failed.append("margin_sufficient")

            # --- Check 7: Not already in position for same index+direction (per user) ---
            index_name = signal.get("index", "")
            direction = signal.get("direction", "")
            if not self._duplicate_position(index_name, direction):
                checks_passed.append("no_duplicate_position")
            else:
                checks_failed.append("no_duplicate_position")

            # --- Check 8: Signal within valid astro window ---
            window_start = signal.get("astro_window_start")
            window_end = signal.get("astro_window_end")
            if window_start and window_end:
                if window_start <= now_ist <= window_end:
                    checks_passed.append("astro_window_valid")
                else:
                    checks_failed.append("astro_window_valid")
            else:
                checks_passed.append("astro_window_valid")

            # --- Check 9: VIX < 20 ---
            max_vix = RISK_RULES.get("max_vix", 20)
            vix_reduce = RISK_RULES.get("vix_reduce_threshold", 18)
            if vix < vix_reduce:
                checks_passed.append("vix_acceptable")
            elif vix_reduce <= vix < max_vix:
                checks_passed.append("vix_acceptable")
                adjusted_quantity = max(1, math.floor(adjusted_quantity * 0.5))
                logger.warning(
                    "[USER:{}] VIX {} in caution zone, lots halved to {}",
                    self.user_id,
                    vix,
                    adjusted_quantity,
                )
            else:
                checks_failed.append("vix_acceptable")

            # --- Check 10: At least 2 of 3 indices agree on direction ---
            bullish_count = signal.get("indices_bullish", 0)
            bearish_count = signal.get("indices_bearish", 0)
            agreement_min = RISK_RULES.get("index_agreement_min", 2)
            if direction.upper() == "CE" and bullish_count >= agreement_min:
                checks_passed.append("index_agreement")
            elif direction.upper() == "PE" and bearish_count >= agreement_min:
                checks_passed.append("index_agreement")
            else:
                checks_failed.append("index_agreement")

            # --- Final verdict ---
            approved = len(checks_failed) == 0
            reason = "ALL_CHECKS_PASSED" if approved else f"BLOCKED: {', '.join(checks_failed)}"

            result = {
                "approved": approved,
                "checks_passed": checks_passed,
                "checks_failed": checks_failed,
                "adjusted_quantity": adjusted_quantity,
                "reason": reason,
            }

            log_fn = logger.info if approved else logger.warning
            log_fn(
                "[USER:{}] pre_trade_check | approved={} | passed={} | failed={} | reason={}",
                self.user_id,
                approved,
                len(checks_passed),
                len(checks_failed),
                reason,
            )
            return result

        except Exception as exc:
            logger.exception("[USER:{}] pre_trade_check crashed: {}", self.user_id, exc)
            return {
                "approved": False,
                "checks_passed": [],
                "checks_failed": ["system_error"],
                "adjusted_quantity": 0,
                "reason": f"SYSTEM_ERROR: {exc}",
            }

    # ------------------------------------------------------------------
    # 2. check_signal (simplified interface for OrderManager)
    # ------------------------------------------------------------------
    def check_signal(self, signal: dict) -> tuple:
        """Simple check_signal interface returning (bool, reason).

        Used by OrderManager.execute_signal().
        """
        result = self.pre_trade_check(signal, self.capital)
        return result["approved"], result["reason"]

    # ------------------------------------------------------------------
    # 3. calculate_position_size
    # ------------------------------------------------------------------
    def calculate_position_size(self, signal: dict) -> int:
        """Calculate position size (lots) for this user's capital."""
        try:
            margin_per_lot = signal.get("margin_required", 0)
            if margin_per_lot <= 0:
                return 1  # default 1 lot

            max_lots = max(1, int(self.capital * 0.02 / margin_per_lot))
            return min(max_lots, RISK_RULES.get("max_lots_per_trade", 3))
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # 4. Daily-limit monitor (per user)
    # ------------------------------------------------------------------
    def check_daily_limits(self) -> dict:
        """
        Query DB for THIS user's today P&L and enforce:
        - Realized loss > daily_loss_limit  -> LOCK this user
        - 3 consecutive losses              -> PAUSE 90 minutes
        - 5 trades done                     -> no new entries
        """
        try:
            now_ist = datetime.now(IST)
            daily_pnl = self._get_today_pnl()
            trade_count = self._get_today_trade_count()
            consecutive_losses = self._get_consecutive_losses()

            loss_limit = -(RISK_RULES.get("daily_loss_limit_pct", 3.0) / 100.0) * self.capital
            max_trades = RISK_RULES.get("max_daily_trades", 5)
            pause_mins = RISK_RULES.get("consecutive_loss_pause_mins", 90)

            status = "ACTIVE"

            if daily_pnl <= loss_limit:
                self._locked = True
                status = "LOCKED"
                logger.error(
                    "[USER:{}] DAILY LOSS LIMIT BREACHED | pnl={:.2f} | limit={:.2f} | LOCKED",
                    self.user_id,
                    daily_pnl,
                    loss_limit,
                )

            if consecutive_losses >= 3 and not self._locked:
                self._pause_until = now_ist + timedelta(minutes=pause_mins)
                status = "PAUSED"
                logger.warning(
                    "[USER:{}] 3 consecutive losses | pausing until {}",
                    self.user_id,
                    self._pause_until.strftime("%H:%M IST"),
                )

            if self._pause_until and now_ist < self._pause_until and not self._locked:
                status = "PAUSED"

            if self._pause_until and now_ist >= self._pause_until:
                self._pause_until = None

            if trade_count >= max_trades and status == "ACTIVE":
                status = "NO_NEW_ENTRIES"
                logger.info("[USER:{}] Max daily trades reached ({})", self.user_id, trade_count)

            result = {
                "user_id": self.user_id,
                "status": status,
                "locked": self._locked,
                "paused_until": self._pause_until.isoformat() if self._pause_until else None,
                "daily_pnl": daily_pnl,
                "trade_count": trade_count,
                "consecutive_losses": consecutive_losses,
                "loss_limit": loss_limit,
                "max_trades": max_trades,
            }

            logger.info(
                "[USER:{}] check_daily_limits | status={} | pnl={:.2f} | trades={}",
                self.user_id,
                status,
                daily_pnl,
                trade_count,
            )
            return result

        except Exception as exc:
            logger.exception("[USER:{}] check_daily_limits crashed: {}", self.user_id, exc)
            return {
                "user_id": self.user_id,
                "status": "ERROR",
                "locked": True,
                "paused_until": None,
                "daily_pnl": 0.0,
                "trade_count": 0,
                "consecutive_losses": 0,
                "loss_limit": 0.0,
                "max_trades": 0,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 5. Emergency exit (per user)
    # ------------------------------------------------------------------
    def emergency_exit_all(self, reason: str) -> None:
        """Market-order exit ALL open positions for THIS user."""
        try:
            logger.critical("[USER:{}] EMERGENCY EXIT TRIGGERED | reason={}", self.user_id, reason)
            self._locked = True

            open_positions = self._get_open_positions()
            if not open_positions:
                logger.info("[USER:{}] No open positions to exit.", self.user_id)
                return

            for pos in open_positions:
                try:
                    order_id = self._place_market_exit(pos)
                    logger.info(
                        "[USER:{}] Emergency exit placed | index={} | direction={} | qty={} | order_id={}",
                        self.user_id,
                        pos.get("index", "?"),
                        pos.get("direction", "?"),
                        pos.get("quantity", 0),
                        order_id,
                    )
                except Exception as order_exc:
                    logger.error(
                        "[USER:{}] Failed to exit position {} : {}",
                        self.user_id,
                        pos.get("position_id", "?"),
                        order_exc,
                    )

            self._log_emergency_exit(reason, open_positions)
            logger.critical(
                "[USER:{}] Emergency exit complete | positions_closed={} | reason={}",
                self.user_id,
                len(open_positions),
                reason,
            )

        except Exception as exc:
            logger.exception("[USER:{}] emergency_exit_all crashed: {}", self.user_id, exc)

    # ------------------------------------------------------------------
    # 6. Reset (new day)
    # ------------------------------------------------------------------
    def reset_daily_state(self) -> None:
        """Reset lock/pause state for a new trading day."""
        self._locked = False
        self._pause_until = None
        logger.info("[USER:{}] Daily risk state reset", self.user_id)

    # ------------------------------------------------------------------
    # Private helpers (DB interaction layer — all filtered by user_id)
    # ------------------------------------------------------------------
    def _get_today_pnl(self) -> float:
        """Return today's realized + unrealized P&L for THIS user."""
        try:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            if hasattr(self.db, "get_daily_pnl"):
                return float(self.db.get_daily_pnl(today_str, user_id=self.user_id) or 0.0)
            if hasattr(self.db, "execute_query"):
                rows = self.db.execute_query(
                    "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE DATE(exit_time) = %s AND user_id = %s",
                    (today_str, self.user_id),
                )
                return float(rows[0][0]) if rows else 0.0
            return 0.0
        except Exception as exc:
            logger.error("[USER:{}] _get_today_pnl failed: {}", self.user_id, exc)
            return 0.0

    def _get_today_trade_count(self) -> int:
        """Return number of trades executed today for THIS user."""
        try:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            if hasattr(self.db, "get_trade_count"):
                return int(self.db.get_trade_count(today_str, user_id=self.user_id) or 0)
            if hasattr(self.db, "execute_query"):
                rows = self.db.execute_query(
                    "SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = %s AND user_id = %s",
                    (today_str, self.user_id),
                )
                return int(rows[0][0]) if rows else 0
            return 0
        except Exception as exc:
            logger.error("[USER:{}] _get_today_trade_count failed: {}", self.user_id, exc)
            return 0

    def _get_consecutive_losses(self) -> int:
        """Return count of consecutive losing trades for THIS user."""
        try:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            if hasattr(self.db, "get_recent_trades"):
                trades = self.db.get_recent_trades(today_str, limit=10, user_id=self.user_id)
            elif hasattr(self.db, "execute_query"):
                trades = self.db.execute_query(
                    "SELECT pnl FROM trades WHERE DATE(exit_time) = %s AND user_id = %s ORDER BY exit_time DESC LIMIT 10",
                    (today_str, self.user_id),
                )
            else:
                return 0

            count = 0
            for trade in trades or []:
                pnl = float(trade[0]) if isinstance(trade, (list, tuple)) else float(trade.get("pnl", 0))
                if pnl < 0:
                    count += 1
                else:
                    break
            return count

        except Exception as exc:
            logger.error("[USER:{}] _get_consecutive_losses failed: {}", self.user_id, exc)
            return 0

    def _duplicate_position(self, index: str, direction: str) -> bool:
        """Check if THIS user already has an open position for same index+direction."""
        try:
            if hasattr(self.db, "has_open_position"):
                return bool(self.db.has_open_position(index, direction, user_id=self.user_id))
            if hasattr(self.db, "execute_query"):
                rows = self.db.execute_query(
                    "SELECT COUNT(*) FROM positions WHERE index_name = %s AND direction = %s AND status = 'OPEN' AND user_id = %s",
                    (index, direction, self.user_id),
                )
                return int(rows[0][0]) > 0 if rows else False
            return False
        except Exception as exc:
            logger.error("[USER:{}] _duplicate_position failed: {}", self.user_id, exc)
            return True  # Block on error to be safe

    def _get_open_positions(self) -> List[dict]:
        """Fetch all open positions for THIS user from DB."""
        try:
            if hasattr(self.db, "get_open_positions"):
                return self.db.get_open_positions(user_id=self.user_id) or []
            if hasattr(self.db, "execute_query"):
                rows = self.db.execute_query(
                    "SELECT position_id, index_name, direction, quantity, entry_price "
                    "FROM positions WHERE status = 'OPEN' AND user_id = %s",
                    (self.user_id,),
                )
                return [
                    {
                        "position_id": r[0],
                        "index": r[1],
                        "direction": r[2],
                        "quantity": r[3],
                        "entry_price": r[4],
                    }
                    for r in (rows or [])
                ]
            return []
        except Exception as exc:
            logger.error("[USER:{}] _get_open_positions failed: {}", self.user_id, exc)
            return []

    def _place_market_exit(self, position: dict) -> str:
        """Place a market exit order. Returns order_id string."""
        if PAPER_TRADE:
            order_id = f"PAPER_EXIT_{self.user_id[:6]}_{position.get('position_id', 'UNK')}_{int(datetime.now(IST).timestamp())}"
            logger.info("[USER:{}] PAPER TRADE exit: {}", self.user_id, order_id)
            return order_id

        if hasattr(self.db, "place_exit_order"):
            return self.db.place_exit_order(position, user_id=self.user_id)

        logger.warning(
            "[USER:{}] No broker connector available for live exit of position {}",
            self.user_id,
            position.get("position_id"),
        )
        return "NO_BROKER"

    def _log_emergency_exit(self, reason: str, positions: List[dict]) -> None:
        """Persist emergency exit event in database."""
        try:
            record = {
                "user_id": self.user_id,
                "timestamp": datetime.now(IST).isoformat(),
                "reason": reason,
                "positions_closed": len(positions),
                "details": positions,
            }
            if hasattr(self.db, "log_event"):
                self.db.log_event("EMERGENCY_EXIT", record)
            elif hasattr(self.db, "execute_query"):
                self.db.execute_query(
                    "INSERT INTO risk_events (event_type, user_id, payload, created_at) VALUES (%s, %s, %s, %s)",
                    ("EMERGENCY_EXIT", self.user_id, str(record), record["timestamp"]),
                )
            logger.info("[USER:{}] Emergency exit logged to DB", self.user_id)
        except Exception as exc:
            logger.error("[USER:{}] _log_emergency_exit failed: {}", self.user_id, exc)
