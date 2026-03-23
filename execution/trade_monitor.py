"""
AstroNifty Trade Monitor — Per-user position monitoring (Multi-User).

Each TradeMonitor instance monitors positions for a SPECIFIC user.
Uses the shared RealtimeHub for price data (no per-user API calls).
Trailing SL, time-based exits, and real-time P&L tracking.
"""

import time
from datetime import datetime, time as dt_time
from typing import Optional

import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
MIS_EXIT_TIME = dt_time(15, 15)
MONITOR_INTERVAL_SECONDS = 5

# Trailing SL thresholds (fraction of target distance)
TRAIL_BREAKEVEN_THRESHOLD = 0.30
TRAIL_HALF_THRESHOLD = 0.60
TRAIL_AGGRESSIVE_THRESHOLD = 0.80


class TradeMonitor:
    """Per-user trade monitor.

    Watches open positions for a SPECIFIC user and triggers SL trails / exits.
    Price data comes from the shared RealtimeHub (or is passed in as a dict).
    """

    def __init__(self, db_manager, order_manager, user_id: str):
        """
        Args:
            db_manager:    Database helper with get_open_trades(user_id), update_trade.
            order_manager: This user's OrderManager for modifying SL, target, closing.
            user_id:       The user whose positions we monitor.
        """
        self.db = db_manager
        self.om = order_manager
        self.user_id = user_id
        self._running = False
        logger.info("TradeMonitor initialised | user={}", user_id)

    # ------------------------------------------------------------------
    # 1. get_open_positions
    # ------------------------------------------------------------------
    def get_open_positions(self) -> list:
        """Return list of open trade dicts for this user."""
        try:
            return self.db.get_open_trades(user_id=self.user_id) or []
        except Exception as exc:
            logger.exception("[USER:{}] get_open_positions failed | {}", self.user_id, exc)
            return []

    # ------------------------------------------------------------------
    # 2. check_all_positions
    # ------------------------------------------------------------------
    def check_all_positions(self, current_prices: dict) -> list:
        """
        Inspect every open position for THIS user against current prices.

        Args:
            current_prices: {symbol: float} mapping of latest prices
                            (typically from shared RealtimeHub cache).

        Returns:
            list of action dicts: [{trade_id, action, details}, ...]
        """
        actions = []
        try:
            open_trades = self.db.get_open_trades(user_id=self.user_id)
            if not open_trades:
                return actions

            for trade in open_trades:
                tid = trade.get("trade_id", "unknown")
                symbol = trade.get("symbol", "")
                price = current_prices.get(symbol)

                if price is None:
                    logger.debug("[USER:{}] No price for {}, skipping", self.user_id, symbol)
                    continue

                # -- update current price in DB for dashboard --
                try:
                    self.db.update_trade(tid, {"current_price": price})
                except Exception:
                    pass

                # -- check SL hit --
                if self._is_sl_hit(trade, price):
                    result = self.om.close_position(tid, reason="SL_HIT")
                    actions.append({"trade_id": tid, "action": "SL_HIT", "details": result, "user_id": self.user_id})
                    logger.warning("[USER:{}] SL HIT | {} @ {}", self.user_id, tid, price)
                    continue

                # -- check target hit --
                if self._is_target_hit(trade, price):
                    result = self.om.close_position(tid, reason="TARGET_HIT")
                    actions.append({"trade_id": tid, "action": "TARGET_HIT", "details": result, "user_id": self.user_id})
                    logger.info("[USER:{}] TARGET HIT | {} @ {}", self.user_id, tid, price)
                    continue

                # -- check time-based exit --
                if self.check_time_exit(trade):
                    result = self.om.close_position(tid, reason="TIME_EXIT_MIS")
                    actions.append({"trade_id": tid, "action": "TIME_EXIT", "details": result, "user_id": self.user_id})
                    logger.info("[USER:{}] TIME EXIT | {}", self.user_id, tid)
                    continue

                # -- trailing SL --
                new_sl = self.trail_stop_loss(trade, price)
                if new_sl is not None:
                    success = self.om.modify_sl(tid, new_sl)
                    if success:
                        actions.append({
                            "trade_id": tid,
                            "action": "SL_TRAILED",
                            "details": {"new_sl": new_sl},
                            "user_id": self.user_id,
                        })
                        logger.info(
                            "[USER:{}] SL trailed | {} new_sl={}",
                            self.user_id,
                            tid,
                            new_sl,
                        )

        except Exception as exc:
            logger.exception("[USER:{}] check_all_positions error | {}", self.user_id, exc)

        return actions

    # ------------------------------------------------------------------
    # 3. trail_stop_loss
    # ------------------------------------------------------------------
    def trail_stop_loss(self, trade: dict, current_price: float) -> Optional[float]:
        """
        Calculate a new SL price based on how far price has moved toward target.

        Rules:
            profit > 80% of target range  ->  SL = entry + 70% of current profit
            profit > 60%                  ->  SL = entry + 50% of current profit
            profit > 30%                  ->  SL = entry  (breakeven)
        """
        try:
            entry = float(trade.get("entry_price", 0))
            sl = float(trade.get("sl_price", 0))
            target = float(trade.get("target_price", 0))
            direction = trade.get("direction", "BUY").upper()

            if direction == "BUY":
                target_distance = target - entry
                current_profit = current_price - entry
            else:
                target_distance = entry - target
                current_profit = entry - current_price

            if target_distance <= 0 or current_profit <= 0:
                return None

            profit_ratio = current_profit / target_distance

            if profit_ratio >= TRAIL_AGGRESSIVE_THRESHOLD:
                trail_amount = current_profit * 0.70
            elif profit_ratio >= TRAIL_HALF_THRESHOLD:
                trail_amount = current_profit * 0.50
            elif profit_ratio >= TRAIL_BREAKEVEN_THRESHOLD:
                trail_amount = 0.0
            else:
                return None

            if direction == "BUY":
                new_sl = round(entry + trail_amount, 2)
                if new_sl <= sl:
                    return None
            else:
                new_sl = round(entry - trail_amount, 2)
                if new_sl >= sl:
                    return None

            new_sl = self._snap_to_tick(new_sl)
            return new_sl

        except Exception as exc:
            logger.exception("[USER:{}] trail_stop_loss error | {}", self.user_id, exc)
            return None

    # ------------------------------------------------------------------
    # 4. check_time_exit
    # ------------------------------------------------------------------
    def check_time_exit(self, trade: dict) -> bool:
        """MIS positions must be squared off by 3:15 PM IST."""
        try:
            product = trade.get("product", "MIS").upper()
            if product != "MIS":
                return False

            now_ist = datetime.now(IST).time()
            return now_ist >= MIS_EXIT_TIME

        except Exception as exc:
            logger.exception("[USER:{}] check_time_exit error | {}", self.user_id, exc)
            return False

    # ------------------------------------------------------------------
    # 5. monitor_loop (standalone mode — typically not used in multi-user)
    # ------------------------------------------------------------------
    def monitor_loop(self, price_provider=None) -> None:
        """
        Continuous monitoring loop for this user.
        In multi-user mode, the scheduler handles monitoring instead.
        This is available for standalone single-user testing.

        Args:
            price_provider: callable that returns {symbol: float} dict.
        """
        self._running = True
        logger.info("[USER:{}] Monitor loop STARTED (interval={}s)", self.user_id, MONITOR_INTERVAL_SECONDS)

        while self._running:
            try:
                now_ist = datetime.now(IST)
                current_time = now_ist.time()

                if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                    logger.debug("[USER:{}] Outside market hours, sleeping 60s", self.user_id)
                    time.sleep(60)
                    continue

                if price_provider:
                    current_prices = price_provider()
                else:
                    current_prices = self._fetch_current_prices()

                if current_prices:
                    actions = self.check_all_positions(current_prices)
                    if actions:
                        logger.info(
                            "[USER:{}] Monitor cycle | {} actions taken",
                            self.user_id,
                            len(actions),
                        )

                time.sleep(MONITOR_INTERVAL_SECONDS)

            except Exception as exc:
                logger.exception("[USER:{}] monitor_loop iteration error | {}", self.user_id, exc)
                time.sleep(MONITOR_INTERVAL_SECONDS)

        logger.info("[USER:{}] Monitor loop STOPPED", self.user_id)

    def stop(self) -> None:
        """Signal the monitor loop to exit gracefully."""
        self._running = False
        logger.info("[USER:{}] Monitor stop requested", self.user_id)

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _is_sl_hit(self, trade: dict, price: float) -> bool:
        """Check whether current price has breached the stop-loss."""
        sl = float(trade.get("sl_price", 0))
        direction = trade.get("direction", "BUY").upper()
        if direction == "BUY":
            return price <= sl
        return price >= sl

    def _is_target_hit(self, trade: dict, price: float) -> bool:
        """Check whether current price has reached the target."""
        target = float(trade.get("target_price", 0))
        direction = trade.get("direction", "BUY").upper()
        if direction == "BUY":
            return price >= target
        return price <= target

    def _fetch_current_prices(self) -> dict:
        """Fallback: fetch LTP via Kite API for this user's open trades.

        In multi-user mode, prices come from the shared RealtimeHub instead.
        """
        prices = {}
        try:
            open_trades = self.db.get_open_trades(user_id=self.user_id)
            if not open_trades:
                return prices

            symbols = list({t.get("symbol") for t in open_trades if t.get("symbol")})
            if not symbols:
                return prices

            # Try the order_manager's kite client
            kite = self.om.kite
            instrument_keys = []
            sym_to_key = {}
            for sym in symbols:
                exchange = "NFO"
                for t in open_trades:
                    if t.get("symbol") == sym:
                        exchange = t.get("exchange", "NFO")
                        break
                key = f"{exchange}:{sym}"
                instrument_keys.append(key)
                sym_to_key[sym] = key

            if hasattr(kite, "ltp"):
                ltp_data = kite.ltp(instrument_keys)
                for sym, key in sym_to_key.items():
                    if key in ltp_data:
                        prices[sym] = ltp_data[key].get("last_price", 0)
            elif hasattr(kite, "quote"):
                quote_data = kite.quote(instrument_keys)
                for sym, key in sym_to_key.items():
                    if key in quote_data:
                        prices[sym] = quote_data[key].get("last_price", 0)

        except Exception as exc:
            logger.exception("[USER:{}] _fetch_current_prices failed | {}", self.user_id, exc)

        return prices

    @staticmethod
    def _snap_to_tick(price: float, tick: float = 0.05) -> float:
        """Round price to nearest tick size."""
        return round(round(price / tick) * tick, 2)
