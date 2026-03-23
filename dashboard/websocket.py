"""
AstroNifty Dashboard — Multi-User WebSocket Manager
Handles per-user WebSocket connections with channel-based real-time data streaming.

Each user_id can have multiple connections (multiple tabs).
Supports:
    - broadcast_to_user(user_id, data) — send to one user's connections
    - broadcast_shared(data) — send shared data to ALL connections
    - broadcast_all(data) — send system messages to ALL connections

Channels:
    prices    — live LTP every second (shared)
    chain     — OI chain updates every 5 seconds (shared)
    greeks    — real-time greeks every 5 seconds (shared)
    scores    — updated composite scores every 60 seconds (per-user)
    signals   — new trading signals immediately (per-user)
    positions — live P&L every second (per-user)
    sectors   — sector pulse every 10 seconds (shared)
    astro     — astro updates every hour (shared)
    health    — module health status (shared)
    trades    — trade log updates (per-user)
    pnl       — P&L summary updates (per-user)
"""

import json
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set

from fastapi import WebSocket
from loguru import logger


# All supported real-time channels
CHANNELS = frozenset([
    "prices",
    "chain",
    "greeks",
    "scores",
    "signals",
    "positions",
    "sectors",
    "astro",
    "health",
    "trades",
    "pnl",
])

# Channels that contain shared data (same for all users)
SHARED_CHANNELS = frozenset(["prices", "chain", "greeks", "sectors", "astro", "health"])

# Channels that contain per-user data
USER_CHANNELS = frozenset(["scores", "signals", "positions", "trades", "pnl"])


class DashboardWebSocket:
    """
    Multi-user WebSocket manager for the AstroNifty dashboard.

    Tracks connections per user_id so that:
    - User-specific data goes only to that user's connections
    - Shared data (astro, OI, prices) goes to everyone
    - System messages (session_expired) go to everyone
    - A single user can have multiple tabs open
    """

    def __init__(self):
        # All connections regardless of user
        self.active_connections: List[WebSocket] = []

        # user_id -> list of WebSocket connections
        self._user_connections: Dict[str, List[WebSocket]] = defaultdict(list)

        # WebSocket -> user_id (reverse lookup)
        self._connection_user: Dict[WebSocket, str] = {}

        # Channel subscriptions per connection
        self._client_channels: Dict[WebSocket, Set[str]] = {}

        # Stats
        self._message_counts: Dict[str, int] = defaultdict(int)
        self._last_broadcast: Dict[str, float] = {}
        self._total_messages_sent: int = 0

        logger.info(
            "Multi-user DashboardWebSocket initialized — {} channels ({} shared, {} per-user)",
            len(CHANNELS),
            len(SHARED_CHANNELS),
            len(USER_CHANNELS),
        )

    async def connect(self, websocket: WebSocket, user_id: str = ""):
        """Accept and register a new WebSocket connection for a specific user."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self._client_channels[websocket] = set(CHANNELS)

        if user_id:
            self._user_connections[user_id].append(websocket)
            self._connection_user[websocket] = user_id

        logger.info(
            "WebSocket connected: {} | user={} | Total: {} | Users online: {}",
            websocket.client,
            user_id[:8] if user_id else "anon",
            len(self.active_connections),
            len(self._user_connections),
        )

    def set_user(self, websocket: WebSocket, user_id: str):
        """Associate (or re-associate) a WebSocket connection with a user_id."""
        # Remove from old user if any
        old_user = self._connection_user.get(websocket)
        if old_user and old_user != user_id:
            if websocket in self._user_connections.get(old_user, []):
                self._user_connections[old_user].remove(websocket)
                if not self._user_connections[old_user]:
                    del self._user_connections[old_user]

        # Add to new user
        self._connection_user[websocket] = user_id
        if websocket not in self._user_connections[user_id]:
            self._user_connections[user_id].append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection from all tracking structures."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        user_id = self._connection_user.pop(websocket, None)
        if user_id and user_id in self._user_connections:
            if websocket in self._user_connections[user_id]:
                self._user_connections[user_id].remove(websocket)
            if not self._user_connections[user_id]:
                del self._user_connections[user_id]

        self._client_channels.pop(websocket, None)

        logger.info(
            "WebSocket disconnected: {} | user={} | Total: {} | Users online: {}",
            websocket.client,
            user_id[:8] if user_id else "anon",
            len(self.active_connections),
            len(self._user_connections),
        )

    def subscribe_client(self, websocket: WebSocket, channels: List[str]):
        """Subscribe a client to specific channels."""
        valid = set(ch for ch in channels if ch in CHANNELS)
        if websocket in self._client_channels:
            self._client_channels[websocket] = valid
            logger.debug("Client {} subscribed to: {}", websocket.client, valid)

    def unsubscribe_client(self, websocket: WebSocket, channels: List[str]):
        """Unsubscribe a client from specific channels."""
        if websocket in self._client_channels:
            self._client_channels[websocket] -= set(channels)
            logger.debug("Client {} unsubscribed from: {}", websocket.client, channels)

    # ------------------------------------------------------------------
    # Broadcasting methods
    # ------------------------------------------------------------------

    async def broadcast_to_user(self, user_id: str, data: dict):
        """Send data to ALL connections belonging to a specific user."""
        connections = self._user_connections.get(user_id, [])
        if not connections:
            return

        channel = data.get("channel") or data.get("type", "")
        payload = json.dumps(data, default=str)
        stale: List[WebSocket] = []

        for conn in connections:
            client_channels = self._client_channels.get(conn, CHANNELS)
            if channel and channel not in client_channels and channel not in ("snapshot", "pong", "system"):
                continue
            try:
                await conn.send_text(payload)
                self._total_messages_sent += 1
            except Exception as e:
                logger.warning("Failed to send to user {} conn {}: {}", user_id[:8], conn.client, e)
                stale.append(conn)

        for conn in stale:
            self.disconnect(conn)

        self._message_counts[f"user:{channel}"] += 1

    async def broadcast_shared(self, data: dict):
        """Send shared data (astro, prices, OI, sectors) to ALL connected clients."""
        if not self.active_connections:
            return

        channel = data.get("channel") or data.get("type", "")
        payload = json.dumps(data, default=str)
        stale: List[WebSocket] = []

        for conn in self.active_connections:
            client_channels = self._client_channels.get(conn, CHANNELS)
            if channel and channel not in client_channels and channel not in ("snapshot", "pong"):
                continue
            try:
                await conn.send_text(payload)
                self._total_messages_sent += 1
            except Exception as e:
                logger.warning("Failed to broadcast shared to {}: {}", conn.client, e)
                stale.append(conn)

        for conn in stale:
            self.disconnect(conn)

        self._message_counts[channel] += 1
        self._last_broadcast[channel] = time.time()

    async def broadcast_all(self, data: dict):
        """Send a system message to ALL connections (e.g., session_expired)."""
        if not self.active_connections:
            return

        payload = json.dumps(data, default=str)
        stale: List[WebSocket] = []

        for conn in self.active_connections:
            try:
                await conn.send_text(payload)
                self._total_messages_sent += 1
            except Exception as e:
                logger.warning("Failed to broadcast_all to {}: {}", conn.client, e)
                stale.append(conn)

        for conn in stale:
            self.disconnect(conn)

        if stale:
            logger.info("Pruned {} stale connections during broadcast_all", len(stale))

    async def broadcast(self, data: dict):
        """Legacy compat: broadcast to all (same as broadcast_shared)."""
        await self.broadcast_shared(data)

    async def broadcast_channel(self, channel: str, data: dict):
        """Convenience: broadcast with channel automatically set."""
        data["channel"] = channel
        await self.broadcast_shared(data)

    async def send_personal(self, websocket: WebSocket, data: dict):
        """Send a JSON message to a single client."""
        try:
            payload = json.dumps(data, default=str)
            await websocket.send_text(payload)
            self._total_messages_sent += 1
        except Exception as e:
            logger.error("Failed to send personal to {}: {}", websocket.client, e)
            self.disconnect(websocket)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_user_id_for_connection(self, websocket: WebSocket) -> Optional[str]:
        """Look up user_id for a given WebSocket connection."""
        return self._connection_user.get(websocket)

    def get_user_connection_count(self, user_id: str) -> int:
        """Number of active connections for a specific user."""
        return len(self._user_connections.get(user_id, []))

    @property
    def connection_count(self) -> int:
        """Total number of active connections."""
        return len(self.active_connections)

    @property
    def online_user_count(self) -> int:
        """Number of unique users currently connected via WebSocket."""
        return len(self._user_connections)

    def get_stats(self) -> dict:
        """Return WebSocket manager statistics."""
        return {
            "active_connections": len(self.active_connections),
            "online_users": len(self._user_connections),
            "total_messages_sent": self._total_messages_sent,
            "messages_per_channel": dict(self._message_counts),
            "last_broadcast_per_channel": {
                ch: round(time.time() - ts, 1)
                for ch, ts in self._last_broadcast.items()
            },
            "available_channels": sorted(CHANNELS),
            "user_connections": {
                uid[:8]: len(conns) for uid, conns in self._user_connections.items()
            },
        }
