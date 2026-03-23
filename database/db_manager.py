"""
AstroNifty Database Manager
Handles all CRUD operations with proper session management, logging, and error handling.
Multi-user support: all per-user methods accept and filter by user_id.
"""

import uuid
import base64
from datetime import datetime, date
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session
from loguru import logger
from config import DATABASE_URL
from database.models import (
    Base, User, Trade, OISnapshot, AstroDaily, WeeklyProbability,
    PatternBacktest, DailyExport, SectorSnapshot,
)


class DBManager:
    """Central database manager for AstroNifty trading system."""

    def __init__(self):
        db_display = DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else DATABASE_URL
        logger.info(f"Initializing DBManager with database: {db_display}")

        # SQLite doesn't support pool_size/max_overflow
        if DATABASE_URL.startswith("sqlite"):
            self.engine = create_engine(
                DATABASE_URL,
                echo=False,
                connect_args={"check_same_thread": False},
            )
        else:
            self.engine = create_engine(
                DATABASE_URL,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
            )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        logger.info("DBManager initialized successfully")

    def _get_session(self) -> Session:
        return self.SessionLocal()

    # ------------------------------------------------------------------
    # Table creation
    # ------------------------------------------------------------------

    def create_tables(self):
        """Create all tables if they don't exist."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables created / verified")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise

    # ==================================================================
    # USER MANAGEMENT
    # ==================================================================

    def create_user(self, api_key: str, api_secret_encrypted: str) -> User | None:
        """
        Register a new user with encrypted Kite credentials.
        Generates a UUID4 user_id. Returns the User object.
        """
        session = self._get_session()
        try:
            user_id = str(uuid.uuid4())
            user = User(
                user_id=user_id,
                api_key=api_key,
                api_secret_hash=api_secret_encrypted,
                capital=500000,
                paper_trade=True,
                is_active=True,
                created_at=datetime.utcnow(),
            )
            session.add(user)
            session.commit()
            logger.info(f"User created: {user_id} | api_key={api_key[:8]}...")
            return user
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create user: {e}")
            return None
        finally:
            session.close()

    def get_user_by_api_key(self, api_key: str) -> User | None:
        """Find a user by their Kite API key."""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.api_key == api_key).first()
            return user
        except Exception as e:
            logger.error(f"Failed to fetch user by api_key: {e}")
            return None
        finally:
            session.close()

    def get_user_by_id(self, user_id: str) -> User | None:
        """Find a user by their UUID user_id."""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            return user
        except Exception as e:
            logger.error(f"Failed to fetch user by id {user_id}: {e}")
            return None
        finally:
            session.close()

    def update_user_session(
        self,
        user_id: str,
        access_token: str,
        zerodha_user_id: str,
        zerodha_user_name: str,
    ) -> bool:
        """Update a user's Kite session after successful login."""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                logger.warning(f"User {user_id} not found for session update")
                return False
            user.access_token = access_token
            user.session_timestamp = datetime.utcnow()
            user.zerodha_user_id = zerodha_user_id
            user.zerodha_user_name = zerodha_user_name
            user.last_login = datetime.utcnow()
            session.commit()
            logger.info(f"User session updated: {user_id} | zerodha={zerodha_user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update user session {user_id}: {e}")
            return False
        finally:
            session.close()

    def update_user_capital(self, user_id: str, capital: float) -> bool:
        """Update a user's trading capital."""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                logger.warning(f"User {user_id} not found for capital update")
                return False
            user.capital = capital
            session.commit()
            logger.info(f"User capital updated: {user_id} -> {capital}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update capital for {user_id}: {e}")
            return False
        finally:
            session.close()

    def get_all_active_users(self) -> list:
        """Return all users with is_active=True."""
        session = self._get_session()
        try:
            users = (
                session.query(User)
                .filter(User.is_active == True)
                .order_by(User.created_at.asc())
                .all()
            )
            logger.debug(f"Active users: {len(users)}")
            return users
        except Exception as e:
            logger.error(f"Failed to fetch active users: {e}")
            return []
        finally:
            session.close()

    def deactivate_user(self, user_id: str) -> bool:
        """Soft-delete a user by setting is_active=False."""
        session = self._get_session()
        try:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                logger.warning(f"User {user_id} not found for deactivation")
                return False
            user.is_active = False
            user.access_token = None
            session.commit()
            logger.info(f"User deactivated: {user_id}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            return False
        finally:
            session.close()

    # ==================================================================
    # TRADE CRUD (per-user)
    # ==================================================================

    def save_trade(self, user_id: str, signal: dict, order_result: dict) -> Trade | None:
        """Persist a new trade from signal + broker order result for a specific user."""
        session = self._get_session()
        try:
            trade = Trade(
                user_id=user_id,
                signal_id=signal.get("signal_id"),
                timestamp=datetime.utcnow(),
                index_name=signal.get("index_name", "NIFTY"),
                signal_type=signal.get("signal_type"),
                strike=signal.get("strike"),
                expiry=signal.get("expiry"),
                instrument_token=signal.get("instrument_token"),
                instrument_symbol=signal.get("instrument_symbol"),
                entry_price=order_result.get("entry_price") or signal.get("entry_price"),
                sl_price=signal.get("sl_price"),
                target_price=signal.get("target_price"),
                quantity=signal.get("quantity", 0),
                total_score=signal.get("total_score"),
                oi_score=signal.get("oi_score"),
                astro_score=signal.get("astro_score"),
                greeks_score=signal.get("greeks_score"),
                confidence=signal.get("confidence"),
                astro_window=signal.get("astro_window"),
                order_id=order_result.get("order_id"),
                sl_order_id=order_result.get("sl_order_id"),
                target_order_id=order_result.get("target_order_id"),
                status=order_result.get("status", "PENDING"),
            )
            session.add(trade)
            session.commit()
            logger.info(f"Trade saved: user={user_id[:8]}.. {trade.signal_id} | {trade.index_name} {trade.signal_type} {trade.strike}")
            return trade
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save trade for user {user_id[:8]}.. {signal.get('signal_id')}: {e}")
            return None
        finally:
            session.close()

    def update_trade(self, trade_id: int, updates: dict) -> bool:
        """Update an existing trade by primary key."""
        session = self._get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                logger.warning(f"Trade {trade_id} not found for update")
                return False
            for key, value in updates.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)
            trade.updated_at = datetime.utcnow()
            session.commit()
            logger.info(f"Trade {trade_id} updated: {list(updates.keys())}")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update trade {trade_id}: {e}")
            return False
        finally:
            session.close()

    def get_todays_trades(self, user_id: str) -> list:
        """Return all trades created today for a specific user."""
        session = self._get_session()
        try:
            today = date.today()
            trades = (
                session.query(Trade)
                .filter(Trade.user_id == user_id)
                .filter(func.date(Trade.created_at) == today)
                .order_by(Trade.created_at.desc())
                .all()
            )
            logger.debug(f"Fetched {len(trades)} trades for user {user_id[:8]}.. on {today}")
            return trades
        except Exception as e:
            logger.error(f"Failed to fetch today's trades for user {user_id[:8]}..: {e}")
            return []
        finally:
            session.close()

    def get_todays_pnl(self, user_id: str) -> float:
        """Sum of PnL for all closed trades today for a specific user."""
        session = self._get_session()
        try:
            today = date.today()
            result = (
                session.query(func.coalesce(func.sum(Trade.pnl), 0.0))
                .filter(Trade.user_id == user_id)
                .filter(func.date(Trade.created_at) == today)
                .filter(Trade.status == "CLOSED")
                .scalar()
            )
            pnl = float(result)
            logger.debug(f"Today's PnL for user {user_id[:8]}..: {pnl}")
            return pnl
        except Exception as e:
            logger.error(f"Failed to calculate today's PnL for user {user_id[:8]}..: {e}")
            return 0.0
        finally:
            session.close()

    def get_open_positions(self, user_id: str) -> list:
        """Return all trades with status OPEN for a specific user."""
        session = self._get_session()
        try:
            positions = (
                session.query(Trade)
                .filter(Trade.user_id == user_id)
                .filter(Trade.status == "OPEN")
                .order_by(Trade.created_at.desc())
                .all()
            )
            logger.debug(f"Open positions for user {user_id[:8]}..: {len(positions)}")
            return positions
        except Exception as e:
            logger.error(f"Failed to fetch open positions for user {user_id[:8]}..: {e}")
            return []
        finally:
            session.close()

    def close_trade(self, user_id: str, trade_id: int, exit_price: float, exit_reason: str) -> bool:
        """Close a trade: compute PnL, set exit fields, mark CLOSED. Validates user ownership."""
        session = self._get_session()
        try:
            trade = (
                session.query(Trade)
                .filter(Trade.id == trade_id)
                .filter(Trade.user_id == user_id)
                .first()
            )
            if not trade:
                logger.warning(f"Trade {trade_id} not found for user {user_id[:8]}.. for closing")
                return False
            if trade.status != "OPEN":
                logger.warning(f"Trade {trade_id} is not OPEN (current: {trade.status})")
                return False

            trade.exit_price = exit_price
            trade.exit_time = datetime.utcnow()
            trade.exit_reason = exit_reason
            trade.status = "CLOSED"

            # PnL calculation
            if trade.entry_price and trade.quantity:
                if trade.signal_type == "CE":
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                else:  # PE — profit when price drops is same direction for buyer
                    trade.pnl = (exit_price - trade.entry_price) * trade.quantity
                if trade.entry_price > 0:
                    trade.pnl_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
                else:
                    trade.pnl_pct = 0.0
            trade.updated_at = datetime.utcnow()

            session.commit()
            logger.info(
                f"Trade {trade_id} CLOSED for user {user_id[:8]}.. | exit={exit_price} reason={exit_reason} "
                f"pnl={trade.pnl} pnl_pct={trade.pnl_pct:.2f}%"
            )
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to close trade {trade_id} for user {user_id[:8]}..: {e}")
            return False
        finally:
            session.close()

    # ==================================================================
    # OI SNAPSHOTS (per-user)
    # ==================================================================

    def save_oi_snapshot(self, user_id: str, chain_data: dict) -> OISnapshot | None:
        """Persist an OI chain snapshot for a specific user."""
        session = self._get_session()
        try:
            snap = OISnapshot(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                index_name=chain_data.get("index_name", "NIFTY"),
                expiry=chain_data.get("expiry"),
                spot_price=chain_data.get("spot_price"),
                max_pain=chain_data.get("max_pain"),
                pcr_overall=chain_data.get("pcr_overall"),
                pcr_weighted=chain_data.get("pcr_weighted"),
                ce_wall_strike=chain_data.get("ce_wall_strike"),
                ce_wall_oi=chain_data.get("ce_wall_oi"),
                pe_wall_strike=chain_data.get("pe_wall_strike"),
                pe_wall_oi=chain_data.get("pe_wall_oi"),
                buildup_pattern=chain_data.get("buildup_pattern"),
                gex_value=chain_data.get("gex_value"),
                gex_interpretation=chain_data.get("gex_interpretation"),
                chain_json=chain_data.get("chain_json"),
            )
            session.add(snap)
            session.commit()
            logger.info(f"OI snapshot saved: user={user_id[:8]}.. {snap.index_name} spot={snap.spot_price} pcr={snap.pcr_overall}")
            return snap
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save OI snapshot for user {user_id[:8]}..: {e}")
            return None
        finally:
            session.close()

    # ==================================================================
    # ASTRO SNAPSHOTS (global — no user_id)
    # ==================================================================

    def save_astro_snapshot(self, astro_data: dict) -> AstroDaily | None:
        """Persist a daily astro snapshot."""
        session = self._get_session()
        try:
            snap = AstroDaily(
                date=astro_data.get("date", date.today()),
                timestamp=datetime.utcnow(),
                jupiter_deg=astro_data.get("jupiter_deg"),
                saturn_deg=astro_data.get("saturn_deg"),
                mars_deg=astro_data.get("mars_deg"),
                venus_deg=astro_data.get("venus_deg"),
                mercury_deg=astro_data.get("mercury_deg"),
                sun_deg=astro_data.get("sun_deg"),
                moon_deg=astro_data.get("moon_deg"),
                rahu_deg=astro_data.get("rahu_deg"),
                ketu_deg=astro_data.get("ketu_deg"),
                jupiter_retro=astro_data.get("jupiter_retro", False),
                saturn_retro=astro_data.get("saturn_retro", False),
                mars_retro=astro_data.get("mars_retro", False),
                venus_retro=astro_data.get("venus_retro", False),
                mercury_retro=astro_data.get("mercury_retro", False),
                nakshatra=astro_data.get("nakshatra"),
                nakshatra_pada=astro_data.get("nakshatra_pada"),
                nakshatra_nature=astro_data.get("nakshatra_nature"),
                tithi=astro_data.get("tithi"),
                tithi_name=astro_data.get("tithi_name"),
                tithi_nature=astro_data.get("tithi_nature"),
                paksha=astro_data.get("paksha"),
                hora_sequence_json=astro_data.get("hora_sequence_json"),
                current_yoga=astro_data.get("current_yoga"),
                active_aspects_json=astro_data.get("active_aspects_json"),
                astro_score=astro_data.get("astro_score"),
                market_bias=astro_data.get("market_bias"),
            )
            session.add(snap)
            session.commit()
            logger.info(f"Astro snapshot saved: {snap.date} score={snap.astro_score} bias={snap.market_bias}")
            return snap
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save astro snapshot: {e}")
            return None
        finally:
            session.close()

    # ==================================================================
    # SECTOR SNAPSHOTS (per-user)
    # ==================================================================

    def save_sector_snapshot(self, user_id: str, sector_data: dict) -> SectorSnapshot | None:
        """Persist a sector snapshot for a specific user."""
        session = self._get_session()
        try:
            snap = SectorSnapshot(
                user_id=user_id,
                timestamp=datetime.utcnow(),
                sector_name=sector_data.get("sector_name"),
                ltp=sector_data.get("ltp"),
                change_pct=sector_data.get("change_pct"),
                volume=sector_data.get("volume"),
                impact_on_nifty=sector_data.get("impact_on_nifty"),
                key_movers_json=sector_data.get("key_movers_json"),
            )
            session.add(snap)
            session.commit()
            logger.info(f"Sector snapshot saved: user={user_id[:8]}.. {snap.sector_name} chg={snap.change_pct}%")
            return snap
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save sector snapshot for user {user_id[:8]}..: {e}")
            return None
        finally:
            session.close()

    # ==================================================================
    # WEEKLY PROBABILITY (global — no user_id)
    # ==================================================================

    def save_weekly_probability(self, forecast: dict) -> WeeklyProbability | None:
        """Persist a weekly probability forecast row."""
        session = self._get_session()
        try:
            row = WeeklyProbability(
                week_start_date=forecast.get("week_start_date"),
                day_of_week=forecast.get("day_of_week"),
                date=forecast.get("date"),
                call_probability=forecast.get("call_probability"),
                put_probability=forecast.get("put_probability"),
                neutral_probability=forecast.get("neutral_probability"),
                predicted_bias=forecast.get("predicted_bias"),
                confidence=forecast.get("confidence"),
                astro_events_json=forecast.get("astro_events_json"),
                model_version=forecast.get("model_version"),
                actual_result=forecast.get("actual_result"),
                prediction_correct=forecast.get("prediction_correct"),
            )
            session.add(row)
            session.commit()
            logger.info(f"Weekly probability saved: {row.date} bias={row.predicted_bias} conf={row.confidence}")
            return row
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save weekly probability: {e}")
            return None
        finally:
            session.close()

    # ==================================================================
    # PATTERN BACKTESTS (global — no user_id)
    # ==================================================================

    def get_pattern_backtests(self, pattern_name: str) -> list:
        """Fetch all backtest rows for a given pattern name."""
        session = self._get_session()
        try:
            results = (
                session.query(PatternBacktest)
                .filter(PatternBacktest.pattern_name == pattern_name)
                .order_by(PatternBacktest.last_updated.desc())
                .all()
            )
            logger.debug(f"Fetched {len(results)} backtests for pattern '{pattern_name}'")
            return results
        except Exception as e:
            logger.error(f"Failed to fetch backtests for '{pattern_name}': {e}")
            return []
        finally:
            session.close()

    # ==================================================================
    # DAILY EXPORTS (per-user)
    # ==================================================================

    def save_daily_export(self, user_id: str, export_info: dict) -> DailyExport | None:
        """Track a daily export job for a specific user."""
        session = self._get_session()
        try:
            row = DailyExport(
                user_id=user_id,
                date=export_info.get("date", date.today()),
                export_type=export_info.get("export_type", "FULL"),
                file_path=export_info.get("file_path"),
                row_count=export_info.get("row_count"),
                status=export_info.get("status", "PENDING"),
            )
            session.add(row)
            session.commit()
            logger.info(f"Daily export saved: user={user_id[:8]}.. {row.date} type={row.export_type} status={row.status}")
            return row
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save daily export for user {user_id[:8]}..: {e}")
            return None
        finally:
            session.close()
