"""
AstroNifty Signal Bus — Internal pub/sub messaging between engine modules.

Events:
    new_signal         — A fresh trading signal was generated
    trade_executed     — An order was placed and filled
    sl_hit             — Stop-loss was triggered on a position
    target_hit         — Target price was reached on a position
    emergency_exit     — Force-close all positions immediately
    score_update       — Composite score changed for an index
    astro_change       — Astro alignment shifted (aspect/transit)
    oi_update          — Open interest data refreshed
    sector_update      — Sector rotation / breadth data refreshed
    tick_update        — Real-time tick batch received from KiteTicker
    price_update       — Live price change for spot/option instruments
    chain_update       — Option chain rebuilt from live OI data
    greeks_update      — Real-time greeks recalculated
    position_pnl_update — Live P&L update for open positions
"""

from __future__ import annotations

import threading
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

VALID_EVENTS = frozenset(
    [
        "new_signal",
        "trade_executed",
        "sl_hit",
        "target_hit",
        "emergency_exit",
        "score_update",
        "astro_change",
        "oi_update",
        "sector_update",
        "tick_update",         # Real-time tick batch received from KiteTicker
        "price_update",        # Live price change for spot/option instruments
        "chain_update",        # Option chain rebuilt from live OI data
        "greeks_update",       # Real-time greeks recalculated
        "position_pnl_update", # Live P&L update for open positions
    ]
)


class SignalBus:
    """Thread-safe publish / subscribe bus for intra-engine communication."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: List[Dict[str, Any]] = []
        self._max_history = 500
        logger.info("SignalBus initialised — {} event types registered", len(VALID_EVENTS))

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------
    def subscribe(self, event: str, callback: Callable) -> None:
        """Register *callback* for *event*.

        Parameters
        ----------
        event : str
            One of the VALID_EVENTS constants.
        callback : callable
            Function with signature ``callback(data: dict) -> None``.
        """
        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Must be one of: {', '.join(sorted(VALID_EVENTS))}"
            )
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback)}")

        with self._lock:
            if callback not in self._subscribers[event]:
                self._subscribers[event].append(callback)
                logger.debug(
                    "Subscribed {} to '{}'",
                    getattr(callback, "__qualname__", repr(callback)),
                    event,
                )

    # ------------------------------------------------------------------
    # Unsubscribe
    # ------------------------------------------------------------------
    def unsubscribe(self, event: str, callback: Callable) -> None:
        """Remove *callback* from *event* subscribers."""
        with self._lock:
            try:
                self._subscribers[event].remove(callback)
                logger.debug(
                    "Unsubscribed {} from '{}'",
                    getattr(callback, "__qualname__", repr(callback)),
                    event,
                )
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish(self, event: str, data: Optional[Dict[str, Any]] = None) -> int:
        """Fire *event* and deliver *data* to every subscriber.

        Parameters
        ----------
        event : str
            One of the VALID_EVENTS constants.
        data : dict, optional
            Payload passed to each callback.  Automatically enriched with
            ``_event`` and ``_ts`` keys.

        Returns
        -------
        int
            Number of callbacks that executed successfully.
        """
        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{event}'. Must be one of: {', '.join(sorted(VALID_EVENTS))}"
            )

        if data is None:
            data = {}

        data["_event"] = event
        data["_ts"] = datetime.now().isoformat()

        # Keep a rolling history for debugging
        self._history.append({"event": event, "ts": data["_ts"], "keys": list(data.keys())})
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

        with self._lock:
            callbacks = list(self._subscribers.get(event, []))

        success_count = 0
        for cb in callbacks:
            try:
                cb(data)
                success_count += 1
            except Exception:
                logger.error(
                    "Subscriber {} failed on '{}': {}",
                    getattr(cb, "__qualname__", repr(cb)),
                    event,
                    traceback.format_exc(),
                )

        logger.debug(
            "Published '{}' — {}/{} subscribers OK",
            event,
            success_count,
            len(callbacks),
        )
        return success_count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def subscriber_count(self, event: str) -> int:
        """Return number of subscribers for *event*."""
        with self._lock:
            return len(self._subscribers.get(event, []))

    def all_subscriber_counts(self) -> Dict[str, int]:
        """Return ``{event: count}`` for every valid event."""
        with self._lock:
            return {ev: len(self._subscribers.get(ev, [])) for ev in sorted(VALID_EVENTS)}

    def recent_events(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return last *n* published events (metadata only, no payloads)."""
        return self._history[-n:]

    def clear(self) -> None:
        """Remove all subscribers and history. Used during shutdown."""
        with self._lock:
            self._subscribers.clear()
        self._history.clear()
        logger.info("SignalBus cleared — all subscribers removed")

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._subscribers.values())
        return f"<SignalBus subscribers={total} events_published={len(self._history)}>"
