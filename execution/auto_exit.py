"""
AstroNifty Auto Exit — Per-user automated exit rules (Multi-User).

Each AutoExit instance handles exits for a SPECIFIC user:
SL, target, time, VIX spikes, and circuit-breaker scenarios.
All DB queries are filtered by user_id.
"""

from datetime import datetime, time as dt_time
from typing import Optional

import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

MIS_EXIT_TIME = dt_time(15, 15)
EOD_CLEANUP_TIME = dt_time(15, 25)
VIX_SPIKE_THRESHOLD = 22.0
VIX_EXTREME_THRESHOLD = 28.0
CIRCUIT_BREAKER_PCT = 10.0


class AutoExit:
    """Per-user rule-based automatic exit engine."""

    def __init__(self, kite_client, db_manager, order_manager, user_id: str):
        """
        Args:
            kite_client:   This user's KiteConnect (or compatible) for market data.
            db_manager:    Database helper (queries filtered by user_id).
            order_manager: This user's OrderManager.
            user_id:       The user this instance serves.
        """
        self.kite = kite_client
        self.db = db_manager
        self.om = order_manager
        self.user_id = user_id
        logger.info("AutoExit initialised | user={}", user_id)

    # ------------------------------------------------------------------
    # 1. check_sl_hit
    # ------------------------------------------------------------------
    def check_sl_hit(self, trade: dict, current_price: float) -> bool:
        """Determine if the stop-loss has been breached."""
        try:
            sl = float(trade.get("sl_price", 0))
            direction = trade.get("direction", "BUY").upper()

            if sl <= 0:
                return False

            if direction == "BUY":
                hit = current_price <= sl
            else:
                hit = current_price >= sl

            if hit:
                logger.warning(
                    "[USER:{}] SL HIT detected | trade={} dir={} sl={} price={}",
                    self.user_id,
                    trade.get("trade_id", "?"),
                    direction,
                    sl,
                    current_price,
                )
            return hit

        except Exception as exc:
            logger.exception("[USER:{}] check_sl_hit error | {}", self.user_id, exc)
            return False

    # ------------------------------------------------------------------
    # 2. check_target_hit
    # ------------------------------------------------------------------
    def check_target_hit(self, trade: dict, current_price: float) -> bool:
        """Determine if the target has been achieved."""
        try:
            target = float(trade.get("target_price", 0))
            direction = trade.get("direction", "BUY").upper()

            if target <= 0:
                return False

            if direction == "BUY":
                hit = current_price >= target
            else:
                hit = current_price <= target

            if hit:
                logger.info(
                    "[USER:{}] TARGET HIT detected | trade={} dir={} target={} price={}",
                    self.user_id,
                    trade.get("trade_id", "?"),
                    direction,
                    target,
                    current_price,
                )
            return hit

        except Exception as exc:
            logger.exception("[USER:{}] check_target_hit error | {}", self.user_id, exc)
            return False

    # ------------------------------------------------------------------
    # 3. handle_sl / handle_target (signal bus callbacks)
    # ------------------------------------------------------------------
    def handle_sl(self, data: dict) -> None:
        """Handle SL hit event from signal bus."""
        trade_id = data.get("trade_id") or data.get("symbol", "unknown")
        logger.warning("[USER:{}] Handling SL hit for {}", self.user_id, trade_id)
        try:
            if data.get("trade_id"):
                self.om.close_position(data["trade_id"], reason="SL_HIT_REALTIME")
        except Exception:
            logger.exception("[USER:{}] handle_sl failed", self.user_id)

    def handle_target(self, data: dict) -> None:
        """Handle target hit event from signal bus."""
        trade_id = data.get("trade_id") or data.get("symbol", "unknown")
        logger.info("[USER:{}] Handling target hit for {}", self.user_id, trade_id)
        try:
            if data.get("trade_id"):
                self.om.close_position(data["trade_id"], reason="TARGET_HIT_REALTIME")
        except Exception:
            logger.exception("[USER:{}] handle_target failed", self.user_id)

    # ------------------------------------------------------------------
    # 4. force_exit_time_based
    # ------------------------------------------------------------------
    def force_exit_time_based(self, trade: dict) -> dict:
        """Exit a MIS position at 3:15 PM IST."""
        tid = trade.get("trade_id", "unknown")
        try:
            product = trade.get("product", "MIS").upper()
            if product != "MIS":
                return {
                    "trade_id": tid,
                    "action": "SKIPPED",
                    "message": "Not MIS, no time exit required",
                }

            now_ist = datetime.now(IST).time()
            if now_ist < MIS_EXIT_TIME:
                return {
                    "trade_id": tid,
                    "action": "SKIPPED",
                    "message": f"Not yet {MIS_EXIT_TIME}, current={now_ist.strftime('%H:%M:%S')}",
                }

            logger.warning(
                "[USER:{}] TIME EXIT triggered | trade={} time={}",
                self.user_id,
                tid,
                now_ist.strftime("%H:%M:%S"),
            )
            result = self.om.close_position(tid, reason="TIME_EXIT_MIS_315PM")
            result["action"] = "TIME_EXIT"
            return result

        except Exception as exc:
            logger.exception("[USER:{}] force_exit_time_based error | {}", self.user_id, exc)
            return {"trade_id": tid, "action": "ERROR", "message": str(exc)}

    # ------------------------------------------------------------------
    # 5. force_exit_vix_spike
    # ------------------------------------------------------------------
    def force_exit_vix_spike(self, trade: dict, vix: float) -> Optional[dict]:
        """Exit if India VIX spikes above the safety threshold."""
        tid = trade.get("trade_id", "unknown")
        try:
            if vix < VIX_SPIKE_THRESHOLD:
                return None

            severity = "EXTREME" if vix >= VIX_EXTREME_THRESHOLD else "HIGH"
            logger.warning(
                "[USER:{}] VIX SPIKE exit | trade={} vix={} severity={}",
                self.user_id,
                tid,
                vix,
                severity,
            )

            reason = f"VIX_SPIKE_{severity}_VIX={round(vix, 2)}"
            result = self.om.close_position(tid, reason=reason)
            result["action"] = "VIX_EXIT"
            result["vix"] = vix
            result["severity"] = severity
            return result

        except Exception as exc:
            logger.exception("[USER:{}] force_exit_vix_spike error | {}", self.user_id, exc)
            return {"trade_id": tid, "action": "ERROR", "message": str(exc)}

    # ------------------------------------------------------------------
    # 6. force_exit_circuit_breaker
    # ------------------------------------------------------------------
    def force_exit_circuit_breaker(self) -> list:
        """Emergency exit ALL of THIS user's positions if market hits circuit breaker."""
        results = []
        try:
            is_circuit, pct_move = self._detect_circuit_breaker()
            if not is_circuit:
                return results

            logger.critical(
                "[USER:{}] CIRCUIT BREAKER detected | nifty_move={}%",
                self.user_id,
                round(pct_move, 2),
            )

            open_trades = self.db.get_open_trades(user_id=self.user_id)
            if not open_trades:
                logger.info("[USER:{}] No open trades to exit for circuit breaker", self.user_id)
                return results

            reason = f"CIRCUIT_BREAKER_NIFTY_MOVE={round(pct_move, 2)}%"
            for trade in open_trades:
                tid = trade.get("trade_id", "unknown")
                try:
                    result = self.om.close_position(tid, reason=reason)
                    result["action"] = "CIRCUIT_BREAKER_EXIT"
                    results.append(result)
                    logger.warning("[USER:{}] Circuit breaker exit | trade={}", self.user_id, tid)
                except Exception as exc:
                    logger.exception(
                        "[USER:{}] Circuit breaker exit failed for {} | {}",
                        self.user_id,
                        tid,
                        exc,
                    )
                    results.append({
                        "trade_id": tid,
                        "action": "ERROR",
                        "message": str(exc),
                    })

        except Exception as exc:
            logger.exception("[USER:{}] force_exit_circuit_breaker error | {}", self.user_id, exc)
            results.append({"action": "ERROR", "message": str(exc)})

        return results

    # ------------------------------------------------------------------
    # 7. end_of_day_cleanup (per-user)
    # ------------------------------------------------------------------
    def end_of_day_cleanup(self) -> dict:
        """
        End-of-day routine for THIS user:
            1. Cancel all pending/open orders
            2. Exit any remaining open positions
            3. Generate EOD summary with P&L

        Returns:
            dict: {cancelled_orders, closed_trades, total_pnl, summary}
        """
        summary = {
            "user_id": self.user_id,
            "cancelled_orders": 0,
            "closed_trades": 0,
            "close_results": [],
            "total_pnl": 0.0,
            "winning_trades": 0,
            "losing_trades": 0,
            "timestamp": datetime.now(IST).isoformat(),
            "errors": [],
        }

        try:
            logger.info("[USER:{}] === EOD CLEANUP STARTED ===", self.user_id)

            # --- 1. Cancel all pending orders for this user ---
            cancelled = self._cancel_all_pending_orders()
            summary["cancelled_orders"] = cancelled
            logger.info("[USER:{}] Cancelled {} pending orders", self.user_id, cancelled)

            # --- 2. Close remaining open positions for this user ---
            open_trades = self.db.get_open_trades(user_id=self.user_id)
            for trade in open_trades:
                tid = trade.get("trade_id", "unknown")
                try:
                    result = self.om.close_position(tid, reason="EOD_CLEANUP")
                    summary["close_results"].append(result)
                    summary["closed_trades"] += 1
                except Exception as exc:
                    error_msg = f"Failed to close {tid}: {exc}"
                    logger.exception("[USER:{}] {}", self.user_id, error_msg)
                    summary["errors"].append(error_msg)

            # --- 3. Compile P&L summary for this user ---
            all_trades_today = self._get_todays_trades()
            for trade in all_trades_today:
                pnl = float(trade.get("pnl", 0))
                summary["total_pnl"] += pnl
                if pnl > 0:
                    summary["winning_trades"] += 1
                elif pnl < 0:
                    summary["losing_trades"] += 1

            summary["total_pnl"] = round(summary["total_pnl"], 2)
            total_trades = summary["winning_trades"] + summary["losing_trades"]
            summary["win_rate"] = (
                round(summary["winning_trades"] / total_trades * 100, 1)
                if total_trades > 0
                else 0.0
            )

            logger.info(
                "[USER:{}] === EOD SUMMARY === | pnl={} wins={} losses={} win_rate={}%",
                self.user_id,
                summary["total_pnl"],
                summary["winning_trades"],
                summary["losing_trades"],
                summary["win_rate"],
            )

        except Exception as exc:
            logger.exception("[USER:{}] end_of_day_cleanup error | {}", self.user_id, exc)
            summary["errors"].append(str(exc))

        return summary

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _detect_circuit_breaker(self) -> tuple:
        """Check if Nifty 50 has moved more than CIRCUIT_BREAKER_PCT from day open."""
        try:
            nifty_key = "NSE:NIFTY 50"

            if hasattr(self.kite, "ohlc"):
                data = self.kite.ohlc([nifty_key])
                if nifty_key in data:
                    ohlc = data[nifty_key].get("ohlc", {})
                    day_open = ohlc.get("open", 0)
                    last_price = data[nifty_key].get("last_price", 0)
                elif hasattr(self.kite, "ltp"):
                    ltp_data = self.kite.ltp([nifty_key])
                    last_price = ltp_data.get(nifty_key, {}).get("last_price", 0)
                    return (False, 0.0)
                else:
                    return (False, 0.0)
            elif hasattr(self.kite, "quote"):
                data = self.kite.quote([nifty_key])
                if nifty_key in data:
                    ohlc = data[nifty_key].get("ohlc", {})
                    day_open = ohlc.get("open", 0)
                    last_price = data[nifty_key].get("last_price", 0)
                else:
                    return (False, 0.0)
            else:
                return (False, 0.0)

            if day_open <= 0:
                return (False, 0.0)

            pct_move = abs((last_price - day_open) / day_open) * 100
            return (pct_move >= CIRCUIT_BREAKER_PCT, pct_move)

        except Exception as exc:
            logger.exception("[USER:{}] _detect_circuit_breaker error | {}", self.user_id, exc)
            return (False, 0.0)

    def _cancel_all_pending_orders(self) -> int:
        """Cancel every open/pending order for THIS user."""
        cancelled = 0
        try:
            if hasattr(self.kite, "orders"):
                orders = self.kite.orders()
                pending_statuses = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
                for order in orders:
                    if order.get("status") in pending_statuses:
                        oid = order.get("order_id")
                        variety = order.get("variety", "regular")
                        try:
                            self.kite.cancel_order(variety=variety, order_id=oid)
                            cancelled += 1
                            logger.info("[USER:{}] EOD cancelled order {}", self.user_id, oid)
                        except Exception as exc:
                            logger.warning(
                                "[USER:{}] Could not cancel order {}: {}",
                                self.user_id,
                                oid,
                                exc,
                            )
            else:
                # Fallback: cancel orders tracked in our DB for this user
                open_trades = self.db.get_open_trades(user_id=self.user_id)
                for trade in open_trades:
                    for oid in trade.get("order_ids", []):
                        try:
                            self.kite.cancel_order(variety="regular", order_id=oid)
                            cancelled += 1
                        except Exception:
                            pass

        except Exception as exc:
            logger.exception("[USER:{}] _cancel_all_pending_orders error | {}", self.user_id, exc)

        return cancelled

    def _get_todays_trades(self) -> list:
        """Retrieve all trades created today for THIS user."""
        try:
            today_str = datetime.now(IST).strftime("%Y-%m-%d")
            if hasattr(self.db, "get_trades_by_date"):
                return self.db.get_trades_by_date(today_str, user_id=self.user_id)
            if hasattr(self.db, "get_all_trades"):
                all_trades = self.db.get_all_trades(user_id=self.user_id)
                return [
                    t for t in all_trades
                    if t.get("created_at", "").startswith(today_str)
                ]
            return []
        except Exception as exc:
            logger.exception("[USER:{}] _get_todays_trades error | {}", self.user_id, exc)
            return []
