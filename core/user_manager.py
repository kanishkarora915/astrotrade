"""
AstroNifty User Manager
========================
Manages unlimited concurrent user sessions.
Each user has their own Kite client, trades, positions, risk manager.
Astro data is SHARED (same for everyone).
OI chain data is SHARED (same market data).
But trades, positions, signals, risk = per-user.

Users are stored in the DB (users table) with UUID4 user_id.
Credentials use base64 encoding (same pattern as existing codebase).
All session mutations are thread-safe via threading.Lock.
"""

import base64
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytz
from loguru import logger

from data.zerodha_client import ZerodhaClient
from execution.order_manager import OrderManager
from execution.trade_monitor import TradeMonitor
from risk.risk_manager import RiskManager
from risk.position_sizer import PositionSizer

IST = pytz.timezone("Asia/Kolkata")

# Kite tokens expire daily around 6:00-6:05 AM IST
KITE_TOKEN_EXPIRY_HOUR = 6
KITE_TOKEN_EXPIRY_MINUTE = 5
# Consider tokens valid for up to 18 hours from generation
KITE_TOKEN_MAX_AGE_HOURS = 18


# ══════════════════════════════════════════════════════════════
# Credential helpers (base64 — same pattern as existing codebase)
# ══════════════════════════════════════════════════════════════

def encrypt_credential(plain: str) -> str:
    """Encode a credential string to base64 for storage."""
    return base64.b64encode(plain.encode("utf-8")).decode("utf-8")


def decrypt_credential(encoded: str) -> str:
    """Decode a base64-encoded credential back to plaintext."""
    return base64.b64decode(encoded.encode("utf-8")).decode("utf-8")


# ══════════════════════════════════════════════════════════════
# UserSession — holds all runtime state for a single user
# ══════════════════════════════════════════════════════════════

class UserSession:
    """Holds all runtime state for a single user."""

    def __init__(self, user_id: str, api_key: str, api_secret: str):
        self.user_id = user_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.kite_client: Optional[ZerodhaClient] = None  # Per-user ZerodhaClient
        self.access_token: Optional[str] = None
        self.is_authenticated: bool = False
        self.capital: float = 500000
        self.paper_trade: bool = True
        self.zerodha_user_id: Optional[str] = None
        self.zerodha_user_name: Optional[str] = None
        self.last_active: float = time.time()

        # Per-user runtime data
        self.scores: dict = {}
        self.signals: list = []
        self.positions: list = []
        self.trades_today: list = []
        self.pnl_summary: dict = {}
        self.oi_data: dict = {}
        self.sector_data: dict = {}

        # Per-user engine components (initialized via init_user_engine)
        self.risk_manager: Optional[RiskManager] = None
        self.order_manager: Optional[OrderManager] = None
        self.trade_monitor: Optional[TradeMonitor] = None
        self.position_sizer: Optional[PositionSizer] = None
        self.realtime_hub = None  # Optional — may share a global hub or be per-user

    def get_login_url(self) -> str:
        """Generate Kite login URL for this user."""
        if not self.kite_client:
            # Create a temporary client just for the login URL
            self.kite_client = ZerodhaClient(
                api_key=self.api_key,
                api_secret=self.api_secret,
            )
        return self.kite_client.get_login_url()

    def get_auth_status(self) -> dict:
        """Return authentication status for this user."""
        return {
            "authenticated": self.is_authenticated,
            "user_id": self.zerodha_user_id,
            "user_name": self.zerodha_user_name,
            "api_key_masked": f"{self.api_key[:4]}****{self.api_key[-2:]}" if len(self.api_key) > 6 else "****",
            "paper_trade": self.paper_trade,
            "capital": self.capital,
        }

    def invalidate(self):
        """Invalidate the session (e.g., at 6:05 AM daily reset)."""
        self.is_authenticated = False
        self.access_token = None
        self.kite_client = None
        self.risk_manager = None
        self.order_manager = None
        self.trade_monitor = None
        self.position_sizer = None
        logger.info("Session invalidated for user {}", self.user_id[:8])

    def touch(self):
        """Update last_active timestamp."""
        self.last_active = time.time()

    def __repr__(self):
        status = "AUTH" if self.is_authenticated else "UNAUTH"
        return f"<UserSession {self.user_id[:8]}.. {self.zerodha_user_id or '?'} [{status}]>"


# ══════════════════════════════════════════════════════════════
# UserManager — manages all user sessions (DB-backed)
# ══════════════════════════════════════════════════════════════

class UserManager:
    """
    Manages unlimited concurrent user sessions.
    Each user has their own Kite client, trades, positions, risk manager.
    Astro data is SHARED (same for everyone).
    OI chain data is SHARED (same market data).
    But trades, positions, signals, risk = per-user.

    All user data is persisted in the database via DBManager.
    Sessions are restored on startup from the users table.
    """

    def __init__(self, db_manager):
        self.db = db_manager
        self.sessions: Dict[str, UserSession] = {}  # user_id -> UserSession
        self._lock = threading.Lock()
        logger.info("UserManager initialized (DB-backed)")

    # ------------------------------------------------------------------
    # Registration / Login
    # ------------------------------------------------------------------

    def register_or_login(self, api_key: str, api_secret: str) -> UserSession:
        """
        If user with this api_key exists, return their session.
        If new, create user in DB and create session.
        Returns UserSession ready for Kite authentication.
        """
        with self._lock:
            # Check if session already exists in memory
            for uid, sess in self.sessions.items():
                if sess.api_key == api_key:
                    sess.touch()
                    logger.info("Existing session found for api_key={}.. user={}..", api_key[:8], uid[:8])
                    return sess

            # Check if user exists in DB
            encrypted_secret = encrypt_credential(api_secret)
            existing_user = self.db.get_user_by_api_key(api_key)

            if existing_user:
                # Returning user — create session from DB record
                user_id = existing_user.user_id
                decrypted_secret = decrypt_credential(existing_user.api_secret_hash)
                session = UserSession(
                    user_id=user_id,
                    api_key=api_key,
                    api_secret=decrypted_secret,
                )
                session.capital = existing_user.capital
                session.paper_trade = existing_user.paper_trade
                session.zerodha_user_id = existing_user.zerodha_user_id
                session.zerodha_user_name = existing_user.zerodha_user_name

                # If there's a valid access token in DB, restore it
                if existing_user.access_token and self._is_token_valid(existing_user.session_timestamp):
                    session.access_token = existing_user.access_token
                    session.is_authenticated = True

                    # Create Kite client with saved token
                    kite_client = ZerodhaClient(
                        api_key=api_key,
                        api_secret=decrypted_secret,
                    )
                    kite_client.kite.set_access_token(existing_user.access_token)
                    session.kite_client = kite_client

                    # Initialize engine components
                    self._init_user_engine_unlocked(session)
                    logger.info("Restored valid token for user {}..", user_id[:8])

                self.sessions[user_id] = session
                logger.info("Session created for returning user {}.. ({})", user_id[:8], existing_user.zerodha_user_id)
                return session
            else:
                # New user — create in DB
                new_user = self.db.create_user(
                    api_key=api_key,
                    api_secret_encrypted=encrypted_secret,
                )
                if not new_user:
                    raise RuntimeError(f"Failed to create user in database for api_key={api_key[:8]}..")

                user_id = new_user.user_id
                session = UserSession(
                    user_id=user_id,
                    api_key=api_key,
                    api_secret=api_secret,
                )
                self.sessions[user_id] = session
                logger.info("New user registered: {}.. api_key={}..", user_id[:8], api_key[:8])
                return session

    # ------------------------------------------------------------------
    # Kite Authentication
    # ------------------------------------------------------------------

    def authenticate_user(self, user_id: str, request_token: str) -> dict:
        """
        Handle Kite callback for a specific user.
        Generate session, save to DB, mark as authenticated.
        Returns dict with session info or error.
        """
        with self._lock:
            session = self.sessions.get(user_id)
            if not session:
                logger.error("No session found for user {}.. during authentication", user_id[:8])
                return {"success": False, "error": "No active session for this user"}

        # Create Kite client for this user (outside lock — network call)
        kite_client = ZerodhaClient(
            api_key=session.api_key,
            api_secret=session.api_secret,
        )

        try:
            # Generate Kite session using request_token
            kite_session = kite_client.kite.generate_session(
                request_token,
                api_secret=session.api_secret,
            )

            access_token = kite_session["access_token"]
            zerodha_user_id = kite_session.get("user_id", "")
            zerodha_user_name = kite_session.get("user_name", "")

            # Set access token on the Kite client
            kite_client.kite.set_access_token(access_token)

            # Try to read capital from profile
            capital = session.capital
            try:
                margins = kite_client.kite.margins()
                equity = margins.get("equity", {})
                live_balance = float(equity.get("available", {}).get("live_balance", 0))
                if live_balance > 0:
                    capital = live_balance
            except Exception:
                pass

            # Update session in memory
            with self._lock:
                session.kite_client = kite_client
                session.access_token = access_token
                session.is_authenticated = True
                session.zerodha_user_id = zerodha_user_id
                session.zerodha_user_name = zerodha_user_name
                session.capital = capital
                session.touch()

                # Initialize per-user engine components
                self._init_user_engine_unlocked(session)

            # Persist to DB
            self.db.update_user_session(
                user_id=user_id,
                access_token=access_token,
                zerodha_user_id=zerodha_user_id,
                zerodha_user_name=zerodha_user_name,
            )
            if capital != session.capital:
                self.db.update_user_capital(user_id, capital)

            logger.success(
                "User authenticated: {}.. | zerodha={} name={}",
                user_id[:8], zerodha_user_id, zerodha_user_name,
            )

            return {
                "success": True,
                "user_id": user_id,
                "zerodha_user_id": zerodha_user_id,
                "zerodha_user_name": zerodha_user_name,
                "capital": capital,
            }

        except Exception as e:
            logger.error("Authentication failed for user {}..: {}", user_id[:8], e)
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # Session lookups
    # ------------------------------------------------------------------

    def get_session(self, user_id: str) -> Optional[UserSession]:
        """Get active session by user_id."""
        with self._lock:
            session = self.sessions.get(user_id)
            if session:
                session.touch()
            return session

    def get_session_by_api_key(self, api_key: str) -> Optional[UserSession]:
        """Find session by API key."""
        with self._lock:
            for uid, sess in self.sessions.items():
                if sess.api_key == api_key:
                    sess.touch()
                    return sess
            return None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def restore_sessions(self):
        """
        On startup, load all active users from DB.
        Restore sessions for users with valid (not expired) access tokens.
        """
        logger.info("Restoring user sessions from database...")
        active_users = self.db.get_all_active_users()
        restored = 0
        expired = 0

        with self._lock:
            for user in active_users:
                user_id = user.user_id
                decrypted_secret = decrypt_credential(user.api_secret_hash)

                session = UserSession(
                    user_id=user_id,
                    api_key=user.api_key,
                    api_secret=decrypted_secret,
                )
                session.capital = user.capital
                session.paper_trade = user.paper_trade
                session.zerodha_user_id = user.zerodha_user_id
                session.zerodha_user_name = user.zerodha_user_name

                # Check if access token is still valid
                if user.access_token and self._is_token_valid(user.session_timestamp):
                    session.access_token = user.access_token
                    session.is_authenticated = True

                    # Create Kite client with saved token
                    try:
                        kite_client = ZerodhaClient(
                            api_key=user.api_key,
                            api_secret=decrypted_secret,
                        )
                        kite_client.kite.set_access_token(user.access_token)
                        session.kite_client = kite_client

                        # Initialize engine components
                        self._init_user_engine_unlocked(session)
                        restored += 1
                    except Exception as exc:
                        logger.warning("Failed to restore Kite client for user {}..: {}", user_id[:8], exc)
                        session.is_authenticated = False
                        expired += 1
                else:
                    expired += 1

                self.sessions[user_id] = session

        logger.info(
            "Session restore complete: {} users loaded, {} authenticated, {} need re-login",
            len(active_users), restored, expired,
        )

    def invalidate_all_sessions(self):
        """
        Called at 6:05 AM IST — all Kite tokens expire.
        Marks all sessions as unauthenticated. Users must re-login.
        """
        with self._lock:
            count = 0
            for uid, sess in self.sessions.items():
                if sess.is_authenticated:
                    sess.invalidate()
                    count += 1

        logger.warning("All {} sessions invalidated (daily Kite token expiry)", count)

    def get_all_authenticated_sessions(self) -> List[UserSession]:
        """Return all users who are currently authenticated."""
        with self._lock:
            return [
                sess for sess in self.sessions.values()
                if sess.is_authenticated
            ]

    def remove_session(self, user_id: str):
        """Remove user session from memory."""
        with self._lock:
            session = self.sessions.pop(user_id, None)
            if session:
                logger.info("Session removed for user {}..", user_id[:8])
            else:
                logger.warning("No session found to remove for user {}..", user_id[:8])

    # ------------------------------------------------------------------
    # Counts & stats
    # ------------------------------------------------------------------

    def get_user_count(self) -> int:
        """Total registered users (in memory)."""
        with self._lock:
            return len(self.sessions)

    def get_active_count(self) -> int:
        """Currently authenticated users."""
        with self._lock:
            return sum(1 for s in self.sessions.values() if s.is_authenticated)

    def get_stats(self) -> dict:
        """Return admin-level stats (no sensitive data)."""
        with self._lock:
            total = len(self.sessions)
            active = sum(1 for s in self.sessions.values() if s.is_authenticated)
            return {
                "total_users": total,
                "active_users": active,
                "inactive_users": total - active,
            }

    # ------------------------------------------------------------------
    # Per-user engine initialization
    # ------------------------------------------------------------------

    def init_user_engine(self, session: UserSession):
        """
        Initialize per-user engine components:
        - RiskManager with user's capital
        - OrderManager with user's Kite client
        - TradeMonitor with user's Kite + OrderManager
        - PositionSizer (stateless, but attached for convenience)

        Thread-safe wrapper.
        """
        with self._lock:
            self._init_user_engine_unlocked(session)

    def _init_user_engine_unlocked(self, session: UserSession):
        """
        Initialize per-user engine components (caller must hold self._lock).
        """
        if not session.kite_client:
            logger.warning("Cannot init engine for user {}.. — no Kite client", session.user_id[:8])
            return

        try:
            # Risk manager — uses user's capital for daily limits
            session.risk_manager = RiskManager(
                db_manager=self.db,
                capital=session.capital,
            )

            # Order manager — places orders via user's Kite client
            session.order_manager = OrderManager(
                kite_client=session.kite_client,
                db_manager=self.db,
                risk_manager=session.risk_manager,
            )

            # Trade monitor — watches open positions, triggers SL/target
            session.trade_monitor = TradeMonitor(
                kite_client=session.kite_client,
                db_manager=self.db,
                order_manager=session.order_manager,
            )

            # Position sizer — stateless utility, attached for convenience
            session.position_sizer = PositionSizer()

            logger.info(
                "Engine initialized for user {}.. ({}) | capital={} paper={}",
                session.user_id[:8], session.zerodha_user_id,
                session.capital, session.paper_trade,
            )

        except Exception as e:
            logger.error("Failed to init engine for user {}..: {}", session.user_id[:8], e)
            # Don't leave partial state
            session.risk_manager = None
            session.order_manager = None
            session.trade_monitor = None
            session.position_sizer = None

    # ------------------------------------------------------------------
    # Per-user data update helpers (called by engine for specific user)
    # ------------------------------------------------------------------

    def update_user_scores(self, user_id: str, data: dict):
        """Update scoring data for a user."""
        session = self.get_session(user_id)
        if session:
            session.scores = data

    def update_user_signals(self, user_id: str, data: list):
        """Update signals list for a user."""
        session = self.get_session(user_id)
        if session:
            session.signals = data

    def update_user_positions(self, user_id: str, data: list):
        """Update positions list for a user."""
        session = self.get_session(user_id)
        if session:
            session.positions = data

    def update_user_trades(self, user_id: str, data: list):
        """Update today's trades for a user."""
        session = self.get_session(user_id)
        if session:
            session.trades_today = data

    def update_user_pnl(self, user_id: str, data: dict):
        """Update PnL summary for a user."""
        session = self.get_session(user_id)
        if session:
            session.pnl_summary = data

    def append_user_signal(self, user_id: str, signal: dict):
        """Append a signal to user's signal list."""
        session = self.get_session(user_id)
        if session:
            session.signals.append(signal)

    def append_user_trade(self, user_id: str, trade: dict):
        """Append a trade to user's daily trades."""
        session = self.get_session(user_id)
        if session:
            session.trades_today.append(trade)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_token_valid(session_timestamp: Optional[datetime]) -> bool:
        """
        Check if a Kite access token is still valid.
        Kite tokens expire daily around 6:00-6:05 AM IST.
        Returns False if timestamp is None or token has expired.
        """
        if session_timestamp is None:
            return False

        now_ist = datetime.now(IST)

        # Make session_timestamp timezone-aware if it isn't
        if session_timestamp.tzinfo is None:
            session_timestamp = IST.localize(session_timestamp)

        # Token too old (> 18 hours)
        age = now_ist - session_timestamp
        if age > timedelta(hours=KITE_TOKEN_MAX_AGE_HOURS):
            return False

        # If token was generated before today's 6:05 AM and it's now past 6:05 AM, it's expired
        today_expiry = now_ist.replace(
            hour=KITE_TOKEN_EXPIRY_HOUR,
            minute=KITE_TOKEN_EXPIRY_MINUTE,
            second=0,
            microsecond=0,
        )
        if session_timestamp < today_expiry <= now_ist:
            return False

        return True
