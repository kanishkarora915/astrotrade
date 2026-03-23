"""
AstroNifty — Telegram Notification Bot (Multi-User).

Per-user Telegram settings. Each user can configure their own
bot_token and chat_id via their settings. If a user hasn't set
Telegram settings, notifications are silently skipped.

Provides both async and sync wrappers for the engine to call.
"""

import asyncio
from datetime import datetime
from typing import Optional

from loguru import logger

try:
    from telegram import Bot
    from telegram.constants import ParseMode
    from telegram.error import TelegramError
except ImportError:
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")
    raise


class TelegramNotifier:
    """
    Per-user Telegram notifier.

    Each user gets their own TelegramNotifier instance (or None if they
    haven't configured Telegram). The bot_token and chat_id come from
    the user's settings_json stored in the User model.
    """

    def __init__(self, bot_token: str, chat_id: str, user_id: str):
        """
        Initialize the Telegram notifier for a specific user.

        Args:
            bot_token: Telegram Bot API token from @BotFather.
            chat_id:   Target chat/group/channel ID for this user.
            user_id:   The user this notifier belongs to.
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.user_id = user_id
        self.bot = Bot(token=self.bot_token)
        self.enabled = True
        logger.info(
            "TelegramNotifier initialized | user={} | chat_id={}",
            user_id,
            chat_id,
        )

    @classmethod
    def from_user_settings(cls, user_id: str, settings: dict) -> Optional["TelegramNotifier"]:
        """Create a TelegramNotifier from a user's settings dict.

        Returns None if Telegram is not configured for this user.

        Args:
            user_id:  The user's ID.
            settings: dict with optional keys: telegram_bot_token, telegram_chat_id.
        """
        bot_token = settings.get("telegram_bot_token")
        chat_id = settings.get("telegram_chat_id")

        if not bot_token or not chat_id:
            logger.debug(
                "User {} has no Telegram settings configured — skipping",
                user_id,
            )
            return None

        return cls(bot_token=bot_token, chat_id=str(chat_id), user_id=user_id)

    # ------------------------------------------------------------------
    # Internal sender
    # ------------------------------------------------------------------
    async def _send(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """Internal sender with error handling and retry."""
        if not self.enabled:
            logger.debug("[USER:{}] Telegram notifications disabled, skipping", self.user_id)
            return False

        for attempt in range(3):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=text,
                    parse_mode=parse_mode,
                    disable_web_page_preview=True,
                )
                logger.debug("[USER:{}] Telegram message sent ({} chars)", self.user_id, len(text))
                return True
            except TelegramError as e:
                logger.warning(
                    "[USER:{}] Telegram send attempt {}/3 failed: {}",
                    self.user_id,
                    attempt + 1,
                    str(e),
                )
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
            except Exception as e:
                logger.error("[USER:{}] Unexpected Telegram error: {}", self.user_id, str(e))
                return False

        logger.error("[USER:{}] Telegram send failed after 3 attempts", self.user_id)
        return False

    def _send_sync(self, text: str, parse_mode: str = ParseMode.HTML) -> bool:
        """Synchronous wrapper for _send — used by engine (non-async context)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — schedule as task
                asyncio.ensure_future(self._send(text, parse_mode))
                return True
            else:
                return loop.run_until_complete(self._send(text, parse_mode))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._send(text, parse_mode))
            finally:
                loop.close()

    # ── Signal Alert ──────────────────────────────────────────────────────

    async def send_signal(self, signal: dict) -> bool:
        """Send a formatted trading signal alert to this user."""
        msg = self.format_signal_message(signal)
        logger.info(
            "[USER:{}] Sending signal alert: {} {} score={}",
            self.user_id,
            signal.get("index"),
            signal.get("direction"),
            signal.get("score"),
        )
        return await self._send(msg)

    def send_signal_sync(self, signal: dict) -> bool:
        """Synchronous wrapper for send_signal."""
        msg = self.format_signal_message(signal)
        return self._send_sync(msg)

    # ── Trade Update ──────────────────────────────────────────────────────

    async def send_trade_update(self, trade: dict, update_type: str) -> bool:
        """Send trade lifecycle updates to this user."""
        type_icons = {
            "ENTRY": "\u2705",
            "SL_HIT": "\U0001F6D1",
            "TARGET_1": "\U0001F3AF",
            "TARGET_2": "\U0001F3AF\U0001F3AF",
            "TRAILING_SL": "\U0001F504",
            "EXIT": "\U0001F6AA",
            "PARTIAL_EXIT": "\u2702\ufe0f",
        }
        icon = type_icons.get(update_type, "\U0001F4CB")
        pnl = trade.get("pnl", 0)
        pnl_icon = "\U0001F7E2" if pnl >= 0 else "\U0001F534"

        msg = (
            f"{icon} <b>TRADE UPDATE: {update_type}</b>\n"
            f"{'=' * 30}\n"
            f"\U0001F4CA <b>{trade.get('instrument', '--')}</b>\n"
            f"\U0001F4C8 Direction: <b>{trade.get('direction', '--')}</b>\n"
            f"\U0001F4B0 Entry: {trade.get('entry_price', '--')}\n"
        )

        if update_type != "ENTRY":
            msg += (
                f"\U0001F3C1 Exit: {trade.get('exit_price', '--')}\n"
                f"{pnl_icon} P&L: <b>{'+' if pnl >= 0 else ''}{pnl:.2f}</b>\n"
            )

        if update_type == "TRAILING_SL":
            msg += f"\U0001F504 New SL: {trade.get('trailing_sl', '--')}\n"

        if trade.get("reason"):
            msg += f"\U0001F4DD Reason: {trade['reason']}\n"

        msg += (
            f"\n\u23F0 {trade.get('time', datetime.now().strftime('%H:%M:%S'))}\n"
            f"{'=' * 30}"
        )

        logger.info("[USER:{}] Sending trade update: {} for {}", self.user_id, update_type, trade.get("instrument"))
        return await self._send(msg)

    def send_trade_update_sync(self, trade: dict, update_type: str) -> bool:
        """Synchronous wrapper for send_trade_update."""
        # Reuse the message building from the async version
        return self._send_sync(self._build_trade_update_msg(trade, update_type))

    def _build_trade_update_msg(self, trade: dict, update_type: str) -> str:
        type_icons = {
            "ENTRY": "\u2705", "SL_HIT": "\U0001F6D1", "TARGET_1": "\U0001F3AF",
            "TARGET_2": "\U0001F3AF\U0001F3AF", "TRAILING_SL": "\U0001F504",
            "EXIT": "\U0001F6AA", "PARTIAL_EXIT": "\u2702\ufe0f",
        }
        icon = type_icons.get(update_type, "\U0001F4CB")
        pnl = trade.get("pnl", 0)
        pnl_icon = "\U0001F7E2" if pnl >= 0 else "\U0001F534"
        msg = (
            f"{icon} <b>TRADE UPDATE: {update_type}</b>\n{'=' * 30}\n"
            f"\U0001F4CA <b>{trade.get('instrument', '--')}</b>\n"
            f"\U0001F4C8 Direction: <b>{trade.get('direction', '--')}</b>\n"
            f"\U0001F4B0 Entry: {trade.get('entry_price', '--')}\n"
        )
        if update_type != "ENTRY":
            msg += (
                f"\U0001F3C1 Exit: {trade.get('exit_price', '--')}\n"
                f"{pnl_icon} P&L: <b>{'+' if pnl >= 0 else ''}{pnl:.2f}</b>\n"
            )
        if trade.get("reason"):
            msg += f"\U0001F4DD Reason: {trade['reason']}\n"
        msg += f"\n\u23F0 {trade.get('time', datetime.now().strftime('%H:%M:%S'))}\n{'=' * 30}"
        return msg

    # ── Daily Summary ─────────────────────────────────────────────────────

    async def send_daily_summary(self, summary: dict) -> bool:
        """Send end-of-day P&L summary to this user."""
        total_pnl = summary.get("net_pnl", summary.get("total_pnl", 0))
        pnl_icon = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        win_rate = summary.get("win_rate", 0)

        msg = (
            f"\U0001F4CA <b>ASTRONIFTY DAILY REPORT</b>\n"
            f"{'=' * 32}\n"
            f"\U0001F4C5 Date: <b>{summary.get('date', datetime.now().strftime('%Y-%m-%d'))}</b>\n\n"
            f"{pnl_icon} <b>NET P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:,.2f}</b>\n"
            f"\U0001F4B5 Gross: {summary.get('gross_pnl', total_pnl):,.2f}\n"
            f"\U0001F4B3 Charges: {summary.get('charges', 0):,.2f}\n\n"
            f"\U0001F4CA <b>TRADE STATS</b>\n"
            f"\u251C Total: {summary.get('total_trades', summary.get('winning_trades', 0) + summary.get('losing_trades', 0))}\n"
            f"\u251C Winners: {summary.get('winning_trades', 0)} \U0001F7E2\n"
            f"\u251C Losers: {summary.get('losing_trades', 0)} \U0001F534\n"
            f"\u251C Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"\u251C Avg Winner: {summary.get('avg_winner', 0):,.2f}\n"
            f"\u251C Avg Loser: {summary.get('avg_loser', 0):,.2f}\n"
            f"\u2514 Risk:Reward: {summary.get('risk_reward', '--')}\n\n"
            f"\U0001F3AF Max Profit: {summary.get('max_profit', 0):,.2f}\n"
            f"\U0001F6D1 Max Loss: {summary.get('max_loss', 0):,.2f}\n\n"
            f"\U0001F4E1 Signals: {summary.get('signals_generated', 0)} generated / "
            f"{summary.get('signals_executed', 0)} executed\n"
            f"\U0001F4B0 Capital Used: {summary.get('capital_used', 0):,.0f}\n"
            f"{'=' * 32}\n"
            f"<i>AstroNifty Engine v2.0 (Multi-User)</i>"
        )

        logger.info("[USER:{}] Sending daily summary: P&L={}", self.user_id, total_pnl)
        return await self._send(msg)

    def send_daily_summary_sync(self, summary: dict) -> bool:
        """Synchronous wrapper for send_daily_summary."""
        total_pnl = summary.get("net_pnl", summary.get("total_pnl", 0))
        # Build message inline (same as async version)
        pnl_icon = "\U0001F7E2" if total_pnl >= 0 else "\U0001F534"
        win_rate = summary.get("win_rate", 0)
        msg = (
            f"\U0001F4CA <b>ASTRONIFTY DAILY REPORT</b>\n"
            f"{'=' * 32}\n"
            f"\U0001F4C5 Date: <b>{summary.get('date', datetime.now().strftime('%Y-%m-%d'))}</b>\n\n"
            f"{pnl_icon} <b>NET P&L: {'+' if total_pnl >= 0 else ''}{total_pnl:,.2f}</b>\n\n"
            f"\U0001F4CA <b>TRADE STATS</b>\n"
            f"\u251C Winners: {summary.get('winning_trades', 0)} | Losers: {summary.get('losing_trades', 0)}\n"
            f"\u2514 Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"{'=' * 32}\n"
            f"<i>AstroNifty Engine v2.0</i>"
        )
        return self._send_sync(msg)

    # ── Pre-market summary ────────────────────────────────────────────────

    def send_pre_market_sync(self, summary: dict) -> bool:
        """Send pre-market summary to this user (sync)."""
        msg = (
            f"\U0001F305 <b>PRE-MARKET SUMMARY</b>\n"
            f"{'=' * 32}\n"
            f"\U0001F4CA Gift Nifty: <b>{summary.get('gift_nifty', '--')}</b>\n"
            f"\U0001F30D Regime: <b>{summary.get('regime', '--')}</b>\n"
            f"\u2728 Astro: {summary.get('astro_today', {}).get('summary', '--')}\n"
            f"\n\u23F0 {summary.get('ts', datetime.now().strftime('%H:%M:%S'))}\n"
            f"{'=' * 32}"
        )
        return self._send_sync(msg)

    # ── Weekly Forecast ───────────────────────────────────────────────────

    async def send_weekly_forecast(self, forecast: dict) -> bool:
        """Send Sunday evening weekly probability forecast to this user."""
        bias = forecast.get("overall_bias", "NEUTRAL")
        bias_icon = "\U0001F7E2" if "bull" in bias.lower() else ("\U0001F534" if "bear" in bias.lower() else "\U0001F7E1")

        msg = (
            f"\U0001F52E <b>ASTRONIFTY WEEKLY FORECAST</b>\n"
            f"{'=' * 34}\n"
            f"\U0001F4C6 {forecast.get('week_start', '--')} to {forecast.get('week_end', '--')}\n\n"
            f"{bias_icon} Overall Bias: <b>{bias}</b> ({forecast.get('confidence', '--')}% conf)\n"
            f"\U0001F4CA NIFTY Range: {forecast.get('nifty_range', '--')}\n"
            f"\U0001F3E6 BANKNIFTY Range: {forecast.get('banknifty_range', '--')}\n\n"
            f"\U0001F4C5 <b>DAY-WISE OUTLOOK</b>\n"
        )

        for day in forecast.get("days", []):
            bull = day.get("bull_pct", 50)
            bear = day.get("bear_pct", 50)
            day_icon = "\U0001F7E2" if bull > bear else ("\U0001F534" if bear > bull else "\U0001F7E1")
            msg += (
                f"\n{day_icon} <b>{day.get('day', '--')}</b>\n"
                f"   Bull: {bull}% | Bear: {bear}%\n"
                f"   Range: {day.get('range', '--')}\n"
                f"   Levels: {day.get('key_levels', '--')}\n"
            )
            if day.get("astro_note"):
                msg += f"   \u2728 {day['astro_note']}\n"

        if forecast.get("key_dates"):
            msg += f"\n\u26A0\ufe0f <b>Key Dates:</b> {', '.join(forecast['key_dates'])}\n"

        msg += (
            f"\n{'=' * 34}\n"
            f"<i>AstroNifty Engine v2.0 | Sunday Forecast</i>"
        )

        logger.info("[USER:{}] Sending weekly forecast: bias={}", self.user_id, bias)
        return await self._send(msg)

    # ── Emergency Alert ───────────────────────────────────────────────────

    async def send_emergency_alert(self, reason: str) -> bool:
        """Send an emergency alert to this user."""
        msg = (
            f"\U0001F6A8\U0001F6A8\U0001F6A8 <b>EMERGENCY ALERT</b> \U0001F6A8\U0001F6A8\U0001F6A8\n"
            f"{'=' * 34}\n\n"
            f"\u26A0\ufe0f <b>REASON:</b> {reason}\n\n"
            f"\U0001F6D1 All your positions may be squared off.\n"
            f"\U0001F512 New entries are BLOCKED.\n\n"
            f"\u23F0 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 34}\n"
            f"<b>IMMEDIATE ACTION REQUIRED</b>"
        )

        logger.critical("[USER:{}] EMERGENCY ALERT: {}", self.user_id, reason)
        return await self._send(msg)

    def send_emergency_alert_sync(self, reason: str) -> bool:
        """Synchronous wrapper for send_emergency_alert."""
        msg = (
            f"\U0001F6A8 <b>EMERGENCY ALERT</b> \U0001F6A8\n"
            f"{'=' * 34}\n"
            f"\u26A0\ufe0f <b>REASON:</b> {reason}\n"
            f"\U0001F6D1 Positions may be squared off.\n"
            f"\u23F0 {datetime.now().strftime('%H:%M:%S')}\n"
            f"{'=' * 34}"
        )
        return self._send_sync(msg)

    # ── Message Formatter ─────────────────────────────────────────────────

    @staticmethod
    def format_signal_message(signal: dict) -> str:
        """Format a trading signal into a Telegram message."""
        direction = (signal.get("direction", "")).upper()
        is_buy = direction == "BUY"
        dir_icon = "\U0001F7E2" if is_buy else "\U0001F534"
        dir_label = "LONG" if is_buy else "SHORT"

        score = signal.get("score", 0)
        confidence = signal.get("confidence", 0)

        filled = min(int(abs(score) / 10), 10)
        bar_color = "\U0001F7E9" if is_buy else "\U0001F7E5"
        strength_bar = bar_color * filled + "\u2B1C" * (10 - filled)

        def bias_icon(b):
            if not b:
                return "\U0001F7E1"
            bl = b.lower()
            return "\U0001F7E2" if "bull" in bl else ("\U0001F534" if "bear" in bl else "\U0001F7E1")

        msg = (
            f"{dir_icon} <b>ASTRONIFTY SIGNAL: {dir_label}</b> {dir_icon}\n"
            f"{'=' * 32}\n\n"
            f"\U0001F4CA <b>Index:</b> {signal.get('index', '--')}\n"
            f"\U0001F3AF <b>Score:</b> {score} / 100\n"
            f"{strength_bar}\n"
            f"\U0001F4AA <b>Confidence:</b> {confidence}%\n\n"
            f"\U0001F4B0 <b>LEVELS</b>\n"
            f"\u251C Entry: <b>{signal.get('entry', '--')}</b>\n"
            f"\u251C SL: <b>{signal.get('sl', '--')}</b>\n"
            f"\u251C Target 1: <b>{signal.get('t1', '--')}</b>\n"
            f"\u2514 Target 2: <b>{signal.get('t2', '--')}</b>\n\n"
            f"\U0001F52D <b>ANALYSIS</b>\n"
            f"\u251C Astro: {bias_icon(signal.get('astro_bias'))} {signal.get('astro_bias', '--')}\n"
            f"\u251C OI: {bias_icon(signal.get('oi_bias'))} {signal.get('oi_bias', '--')}\n"
            f"\u2514 Sector: {bias_icon(signal.get('sector_bias'))} {signal.get('sector_bias', '--')}\n\n"
        )

        if signal.get("reason"):
            msg += f"\U0001F4DD <b>Reason:</b> {signal['reason']}\n\n"

        msg += (
            f"\u23F0 {signal.get('time', datetime.now().strftime('%H:%M:%S'))}\n"
            f"{'=' * 32}\n"
            f"<i>AstroNifty Engine v2.0</i>"
        )

        return msg
