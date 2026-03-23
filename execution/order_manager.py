"""
AstroNifty Order Manager — Per-user execution layer (Multi-User).

Each OrderManager instance is bound to a SPECIFIC user's Kite client.
All orders are placed via that user's broker connection.
All DB operations include user_id for isolation.
"""

import uuid
import time
from datetime import datetime
from typing import Optional

import pytz
from loguru import logger

try:
    from config import PAPER_TRADE, INDICES
except ImportError:
    PAPER_TRADE = True
    INDICES = {
        "NIFTY": {"lot_size": 25, "exchange": "NFO", "tick_size": 0.05},
        "BANKNIFTY": {"lot_size": 15, "exchange": "NFO", "tick_size": 0.05},
        "FINNIFTY": {"lot_size": 25, "exchange": "NFO", "tick_size": 0.05},
    }

IST = pytz.timezone("Asia/Kolkata")


class OrderManager:
    """Per-user order manager.

    Bridges astro/technical signals to live Kite orders for a SPECIFIC user.
    Each user gets their own OrderManager instance with their own kite_client.
    All database operations are scoped to user_id.
    """

    _paper_order_counter: int = 0

    def __init__(self, kite_client, db_manager, risk_manager, user_id: str):
        """
        Args:
            kite_client:  This user's KiteConnect instance.
            db_manager:   Database helper (shared, but queries filtered by user_id).
            risk_manager: This user's RiskManager instance.
            user_id:      Unique identifier for this user.
        """
        self.kite = kite_client
        self.db = db_manager
        self.risk = risk_manager
        self.user_id = user_id
        logger.info(
            "OrderManager initialised | user={} | paper_trade={}",
            user_id,
            PAPER_TRADE,
        )

    # ------------------------------------------------------------------
    # 1. execute_signal — top-level entry point
    # ------------------------------------------------------------------
    def execute_signal(self, signal: dict) -> dict:
        """
        Full flow: risk check -> position size -> place bracket order -> persist.

        All DB operations include self.user_id for isolation.
        Orders placed via THIS user's Kite client.

        Args:
            signal: dict with keys:
                symbol, direction, entry_price, sl_price, target_price,
                strategy, signal_source, product, and optional: reason,
                confidence, expiry, user_id.

        Returns:
            dict: {trade_id, order_ids, status, message}
        """
        trade_id = self._generate_trade_id(signal)
        result = {"trade_id": trade_id, "order_ids": [], "status": "REJECTED", "message": ""}

        try:
            # --- risk gate (per-user) ---
            allowed, reason = self.risk.check_signal(signal)
            if not allowed:
                result["message"] = f"Risk rejected: {reason}"
                logger.warning(
                    "[USER:{}] Signal REJECTED by risk | {}",
                    self.user_id,
                    reason,
                )
                self._save_trade_record(trade_id, signal, "REJECTED", reason=reason)
                return result

            # --- position sizing (per-user capital) ---
            lots = self.risk.calculate_position_size(signal)
            if lots <= 0:
                result["message"] = "Position size zero after risk calc"
                logger.warning("[USER:{}] Position size is 0 - skipping", self.user_id)
                self._save_trade_record(trade_id, signal, "REJECTED", reason="zero_size")
                return result

            index_key = self._resolve_index(signal.get("symbol", ""))
            lot_size = INDICES.get(index_key, {}).get("lot_size", 25)
            quantity = lots * lot_size

            # --- persist BEFORE placing (so we never lose a record) ---
            trade_record = self._build_trade_record(trade_id, signal, quantity, lots)
            self.db.save_trade(trade_record)
            logger.info(
                "[USER:{}] Trade saved pre-execution | id={} qty={}",
                self.user_id,
                trade_id,
                quantity,
            )

            # --- place orders via THIS user's Kite client ---
            order_result = self.place_entry_with_protection(
                signal, trade_id=trade_id, quantity=quantity
            )

            result["order_ids"] = order_result.get("order_ids", [])
            result["status"] = order_result.get("status", "ERROR")
            result["message"] = order_result.get("message", "")

            # --- update DB with order IDs ---
            self.db.update_trade(trade_id, {
                "order_ids": result["order_ids"],
                "status": result["status"],
                "executed_at": datetime.now(IST).isoformat(),
            })

            logger.info(
                "[USER:{}] Signal executed | id={} status={} orders={}",
                self.user_id,
                trade_id,
                result["status"],
                result["order_ids"],
            )

        except Exception as exc:
            logger.exception("[USER:{}] execute_signal FAILED | {}", self.user_id, exc)
            result["status"] = "ERROR"
            result["message"] = str(exc)
            try:
                self.db.update_trade(trade_id, {"status": "ERROR", "error": str(exc)})
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # 2. place_entry_with_protection
    # ------------------------------------------------------------------
    def place_entry_with_protection(
        self,
        signal: dict,
        trade_id: Optional[str] = None,
        quantity: Optional[int] = None,
    ) -> dict:
        """
        Place entry + SL + target via THIS user's Kite client.
        Tries bracket order first; falls back to 3 separate linked orders.
        """
        trade_id = trade_id or self._generate_trade_id(signal)
        if quantity is None:
            index_key = self._resolve_index(signal.get("symbol", ""))
            lot_size = INDICES.get(index_key, {}).get("lot_size", 25)
            quantity = lot_size

        entry_price = float(signal["entry_price"])
        sl_price = float(signal["sl_price"])
        target_price = float(signal["target_price"])
        direction = signal["direction"].upper()
        symbol = signal["symbol"]
        product = signal.get("product", "MIS")
        exchange = signal.get("exchange", "NFO")

        transaction = "BUY" if direction == "BUY" else "SELL"
        sl_transaction = "SELL" if direction == "BUY" else "BUY"

        sl_points = abs(entry_price - sl_price)
        target_points = abs(target_price - entry_price)

        if PAPER_TRADE:
            return self._paper_place(trade_id, signal, quantity)

        # --- try bracket order (via THIS user's kite) ---
        try:
            if hasattr(self.kite, "place_order"):
                order_id = self.kite.place_order(
                    variety="bo",
                    exchange=exchange,
                    tradingsymbol=symbol,
                    transaction_type=transaction,
                    quantity=quantity,
                    order_type="LIMIT",
                    price=entry_price,
                    squareoff=round(target_points, 2),
                    stoploss=round(sl_points, 2),
                    product=product,
                    tag=trade_id[:20],
                )
                logger.info(
                    "[USER:{}] Bracket order placed | order_id={}",
                    self.user_id,
                    order_id,
                )
                return {
                    "order_ids": [order_id],
                    "status": "OPEN",
                    "message": "Bracket order placed",
                    "order_type": "bracket",
                }
        except Exception as bo_err:
            logger.warning(
                "[USER:{}] Bracket order failed, falling back | {}",
                self.user_id,
                bo_err,
            )

        # --- fallback: 3 separate orders ---
        order_ids = []
        try:
            entry_oid = self.kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction,
                quantity=quantity,
                order_type="LIMIT",
                price=entry_price,
                product=product,
                tag=trade_id[:20],
            )
            order_ids.append(entry_oid)
            logger.info("[USER:{}] Entry order placed | oid={}", self.user_id, entry_oid)

            sl_trigger = sl_price
            sl_oid = self.kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=sl_transaction,
                quantity=quantity,
                order_type="SL",
                trigger_price=sl_trigger,
                price=round(sl_trigger - 0.5 if direction == "BUY" else sl_trigger + 0.5, 2),
                product=product,
                tag=trade_id[:20],
            )
            order_ids.append(sl_oid)
            logger.info("[USER:{}] SL order placed | oid={}", self.user_id, sl_oid)

            target_oid = self.kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=sl_transaction,
                quantity=quantity,
                order_type="LIMIT",
                price=target_price,
                product=product,
                tag=trade_id[:20],
            )
            order_ids.append(target_oid)
            logger.info("[USER:{}] Target order placed | oid={}", self.user_id, target_oid)

            return {
                "order_ids": order_ids,
                "status": "OPEN",
                "message": "3-leg orders placed",
                "order_type": "separate",
            }

        except Exception as exc:
            logger.exception(
                "[USER:{}] Separate-leg placement failed | {}",
                self.user_id,
                exc,
            )
            for oid in order_ids:
                try:
                    self.kite.cancel_order(variety="regular", order_id=oid)
                    logger.info("[USER:{}] Rolled back order {}", self.user_id, oid)
                except Exception:
                    pass
            return {
                "order_ids": order_ids,
                "status": "ERROR",
                "message": f"Leg placement failed: {exc}",
                "order_type": "separate",
            }

    # ------------------------------------------------------------------
    # 3. modify_sl
    # ------------------------------------------------------------------
    def modify_sl(self, trade_id: str, new_sl: float) -> bool:
        """Trail the stop-loss order for a given trade."""
        try:
            trade = self.db.get_trade(trade_id)
            if not trade:
                logger.warning("[USER:{}] modify_sl: trade not found | {}", self.user_id, trade_id)
                return False

            # Verify this trade belongs to this user
            if trade.get("user_id") != self.user_id:
                logger.warning(
                    "[USER:{}] modify_sl: trade {} belongs to different user",
                    self.user_id,
                    trade_id,
                )
                return False

            if PAPER_TRADE:
                self.db.update_trade(trade_id, {"sl_price": new_sl, "sl_modified_at": datetime.now(IST).isoformat()})
                logger.info("[USER:{}] Paper SL modified | trade={} new_sl={}", self.user_id, trade_id, new_sl)
                return True

            sl_order_id = self._get_sl_order_id(trade)
            if not sl_order_id:
                logger.warning("[USER:{}] No SL order_id found for trade {}", self.user_id, trade_id)
                return False

            direction = trade.get("direction", "BUY").upper()
            sl_limit = round(new_sl - 0.5 if direction == "BUY" else new_sl + 0.5, 2)

            self.kite.modify_order(
                variety="regular",
                order_id=sl_order_id,
                trigger_price=new_sl,
                price=sl_limit,
            )

            self.db.update_trade(trade_id, {
                "sl_price": new_sl,
                "sl_modified_at": datetime.now(IST).isoformat(),
            })
            logger.info("[USER:{}] SL modified | trade={} new_sl={}", self.user_id, trade_id, new_sl)
            return True

        except Exception as exc:
            logger.exception("[USER:{}] modify_sl failed | {}", self.user_id, exc)
            return False

    # ------------------------------------------------------------------
    # 4. modify_target
    # ------------------------------------------------------------------
    def modify_target(self, trade_id: str, new_target: float) -> bool:
        """Modify the target order for a given trade."""
        try:
            trade = self.db.get_trade(trade_id)
            if not trade:
                logger.warning("[USER:{}] modify_target: trade not found | {}", self.user_id, trade_id)
                return False

            if trade.get("user_id") != self.user_id:
                logger.warning(
                    "[USER:{}] modify_target: trade {} belongs to different user",
                    self.user_id,
                    trade_id,
                )
                return False

            if PAPER_TRADE:
                self.db.update_trade(trade_id, {
                    "target_price": new_target,
                    "target_modified_at": datetime.now(IST).isoformat(),
                })
                logger.info("[USER:{}] Paper target modified | trade={} new_tgt={}", self.user_id, trade_id, new_target)
                return True

            target_order_id = self._get_target_order_id(trade)
            if not target_order_id:
                logger.warning("[USER:{}] No target order_id for trade {}", self.user_id, trade_id)
                return False

            self.kite.modify_order(
                variety="regular",
                order_id=target_order_id,
                price=new_target,
            )

            self.db.update_trade(trade_id, {
                "target_price": new_target,
                "target_modified_at": datetime.now(IST).isoformat(),
            })
            logger.info("[USER:{}] Target modified | trade={} new_tgt={}", self.user_id, trade_id, new_target)
            return True

        except Exception as exc:
            logger.exception("[USER:{}] modify_target failed | {}", self.user_id, exc)
            return False

    # ------------------------------------------------------------------
    # 5. close_position
    # ------------------------------------------------------------------
    def close_position(self, trade_id: str, reason: str) -> dict:
        """Market-order exit for the given trade (this user only)."""
        result = {"trade_id": trade_id, "status": "ERROR", "message": "", "exit_price": None}

        try:
            trade = self.db.get_trade(trade_id)
            if not trade:
                result["message"] = "Trade not found"
                return result

            if trade.get("user_id") != self.user_id:
                result["message"] = "Trade belongs to different user"
                return result

            if trade.get("status") in ("CLOSED", "CANCELLED"):
                result["status"] = "ALREADY_CLOSED"
                result["message"] = "Trade already closed"
                return result

            direction = trade.get("direction", "BUY").upper()
            exit_transaction = "SELL" if direction == "BUY" else "BUY"
            quantity = trade.get("quantity", 0)
            symbol = trade.get("symbol", "")
            exchange = trade.get("exchange", "NFO")
            product = trade.get("product", "MIS")

            self._cancel_pending_orders(trade)

            if PAPER_TRADE:
                exit_price = trade.get("current_price", trade.get("entry_price", 0))
                pnl = self._calculate_pnl(trade, exit_price)
                self.db.update_trade(trade_id, {
                    "status": "CLOSED",
                    "exit_price": exit_price,
                    "exit_reason": reason,
                    "pnl": pnl,
                    "closed_at": datetime.now(IST).isoformat(),
                })
                result["status"] = "CLOSED"
                result["exit_price"] = exit_price
                result["message"] = f"Paper closed: {reason}"
                result["pnl"] = pnl
                logger.info(
                    "[USER:{}] Paper position closed | {} pnl={} reason={}",
                    self.user_id,
                    trade_id,
                    pnl,
                    reason,
                )
                return result

            # --- live market exit via THIS user's Kite ---
            exit_oid = self.kite.place_order(
                variety="regular",
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=exit_transaction,
                quantity=quantity,
                order_type="MARKET",
                product=product,
                tag=trade_id[:20],
            )
            logger.info("[USER:{}] Exit order placed | oid={} reason={}", self.user_id, exit_oid, reason)

            self.db.update_trade(trade_id, {
                "status": "CLOSING",
                "exit_order_id": exit_oid,
                "exit_reason": reason,
                "closed_at": datetime.now(IST).isoformat(),
            })
            result["status"] = "CLOSING"
            result["message"] = f"Exit order placed: {reason}"
            result["order_ids"] = [exit_oid]

        except Exception as exc:
            logger.exception("[USER:{}] close_position failed | {}", self.user_id, exc)
            result["message"] = str(exc)

        return result

    # ------------------------------------------------------------------
    # 6. close_all_positions
    # ------------------------------------------------------------------
    def close_all_positions(self, reason: str) -> list:
        """Emergency exit: close every open position for THIS user only."""
        results = []
        try:
            open_trades = self.db.get_open_trades(user_id=self.user_id)
            logger.warning(
                "[USER:{}] CLOSE ALL triggered | count={} reason={}",
                self.user_id,
                len(open_trades),
                reason,
            )
            for trade in open_trades:
                tid = trade.get("trade_id", "unknown")
                res = self.close_position(tid, reason=reason)
                results.append(res)
        except Exception as exc:
            logger.exception("[USER:{}] close_all_positions failed | {}", self.user_id, exc)
            results.append({"status": "ERROR", "message": str(exc)})
        return results

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _generate_trade_id(self, signal: dict) -> str:
        """Unique trade ID: ASTRO-<user>-<symbol>-<timestamp>-<short uuid>."""
        user_short = self.user_id[:6] if self.user_id else "SYS"
        sym = signal.get("symbol", "UNK")[:10]
        ts = datetime.now(IST).strftime("%Y%m%d%H%M%S")
        short = uuid.uuid4().hex[:6]
        return f"ASTRO-{user_short}-{sym}-{ts}-{short}"

    def _resolve_index(self, symbol: str) -> str:
        """Map a tradingsymbol back to its parent index key."""
        upper = symbol.upper()
        if "BANKNIFTY" in upper:
            return "BANKNIFTY"
        if "FINNIFTY" in upper:
            return "FINNIFTY"
        return "NIFTY"

    def _build_trade_record(self, trade_id: str, signal: dict, quantity: int, lots: int) -> dict:
        return {
            "trade_id": trade_id,
            "user_id": self.user_id,
            "symbol": signal.get("symbol"),
            "direction": signal.get("direction", "BUY").upper(),
            "entry_price": float(signal.get("entry_price", 0)),
            "sl_price": float(signal.get("sl_price", 0)),
            "target_price": float(signal.get("target_price", 0)),
            "quantity": quantity,
            "lots": lots,
            "product": signal.get("product", "MIS"),
            "exchange": signal.get("exchange", "NFO"),
            "strategy": signal.get("strategy", "astro"),
            "signal_source": signal.get("signal_source", "unknown"),
            "confidence": signal.get("confidence"),
            "reason": signal.get("reason"),
            "status": "PENDING",
            "order_ids": [],
            "created_at": datetime.now(IST).isoformat(),
        }

    def _save_trade_record(
        self,
        trade_id: str,
        signal: dict,
        status: str,
        reason: str = "",
    ) -> None:
        """Persist a trade record (including rejected ones) with user_id."""
        try:
            record = self._build_trade_record(trade_id, signal, 0, 0)
            record["status"] = status
            record["reject_reason"] = reason
            self.db.save_trade(record)
        except Exception:
            logger.exception("[USER:{}] Failed to save trade record", self.user_id)

    def _paper_place(self, trade_id: str, signal: dict, quantity: int) -> dict:
        """Simulate order placement for paper trading."""
        OrderManager._paper_order_counter += 1
        fake_oid = f"PAPER-{self.user_id[:6]}-{OrderManager._paper_order_counter}"
        logger.info(
            "[USER:{}] PAPER order placed | tid={} sym={} dir={} qty={}",
            self.user_id,
            trade_id,
            signal.get("symbol"),
            signal.get("direction"),
            quantity,
        )
        return {
            "order_ids": [fake_oid],
            "status": "OPEN",
            "message": "Paper trade executed",
            "order_type": "paper",
        }

    def _get_sl_order_id(self, trade: dict) -> Optional[str]:
        order_ids = trade.get("order_ids", [])
        if len(order_ids) >= 2:
            return order_ids[1]
        return trade.get("sl_order_id")

    def _get_target_order_id(self, trade: dict) -> Optional[str]:
        order_ids = trade.get("order_ids", [])
        if len(order_ids) >= 3:
            return order_ids[2]
        return trade.get("target_order_id")

    def _cancel_pending_orders(self, trade: dict) -> None:
        """Best-effort cancel of SL and target legs."""
        if PAPER_TRADE:
            return
        for oid in trade.get("order_ids", [])[1:]:
            try:
                self.kite.cancel_order(variety="regular", order_id=oid)
                logger.info("[USER:{}] Cancelled pending order {}", self.user_id, oid)
            except Exception as exc:
                logger.warning("[USER:{}] Could not cancel order {}: {}", self.user_id, oid, exc)

    def _calculate_pnl(self, trade: dict, exit_price: float) -> float:
        entry = float(trade.get("entry_price", 0))
        qty = int(trade.get("quantity", 0))
        direction = trade.get("direction", "BUY").upper()
        if direction == "BUY":
            return round((exit_price - entry) * qty, 2)
        return round((entry - exit_price) * qty, 2)
