"""
AstroNifty Database Models
SQLAlchemy 2.0 ORM - All tables for users, trade, OI, astro, sector, backtest, export tracking.
Multi-user support: User model + user_id on all per-user tables.
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Date, Text, JSON,
    Enum, create_engine
)
from sqlalchemy.orm import declarative_base
from config import DATABASE_URL

Base = declarative_base()


# ══════════════════════════════════════════════════════════════
# USER MODEL — One row per registered user
# ══════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)  # UUID
    api_key = Column(String(128), nullable=False)  # Kite API key (encrypted)
    api_secret_hash = Column(String(256), nullable=False)  # Kite API secret (encrypted)
    access_token = Column(String(256), nullable=True)  # Current session token
    session_timestamp = Column(DateTime, nullable=True)  # When token was generated
    zerodha_user_id = Column(String(32), nullable=True)  # Zerodha login ID
    zerodha_user_name = Column(String(128), nullable=True)
    capital = Column(Float, default=500000)  # User's trading capital
    paper_trade = Column(Boolean, default=True)  # Paper or live mode
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    settings_json = Column(JSON, nullable=True)  # Per-user settings override

    def __repr__(self):
        return f"<User {self.user_id} zerodha={self.zerodha_user_id} active={self.is_active}>"


# ══════════════════════════════════════════════════════════════
# PER-USER TABLES — All have user_id column
# ══════════════════════════════════════════════════════════════

class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    signal_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    index_name = Column(String(20), nullable=False)  # NIFTY / BANKNIFTY / FINNIFTY
    signal_type = Column(String(10), nullable=False)  # CE / PE
    strike = Column(Float, nullable=False)
    expiry = Column(Date, nullable=False)
    instrument_token = Column(String(32), nullable=True)
    instrument_symbol = Column(String(64), nullable=True)
    entry_price = Column(Float, nullable=True)
    sl_price = Column(Float, nullable=True)
    target_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=False, default=0)
    total_score = Column(Float, nullable=True)
    oi_score = Column(Float, nullable=True)
    astro_score = Column(Float, nullable=True)
    greeks_score = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    astro_window = Column(String(32), nullable=True)
    order_id = Column(String(64), nullable=True)
    sl_order_id = Column(String(64), nullable=True)
    target_order_id = Column(String(64), nullable=True)
    status = Column(
        String(16),  # PENDING / OPEN / CLOSED / CANCELLED
        nullable=False,
        default="PENDING",
    )
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(64), nullable=True)  # SL_HIT / TARGET_HIT / MANUAL / EXPIRY
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Trade {self.signal_id} {self.index_name} {self.signal_type} {self.strike} {self.status}>"


class OISnapshot(Base):
    __tablename__ = "oi_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    index_name = Column(String(20), nullable=False)
    expiry = Column(Date, nullable=True)
    spot_price = Column(Float, nullable=True)
    max_pain = Column(Float, nullable=True)
    pcr_overall = Column(Float, nullable=True)
    pcr_weighted = Column(Float, nullable=True)
    ce_wall_strike = Column(Float, nullable=True)
    ce_wall_oi = Column(Float, nullable=True)
    pe_wall_strike = Column(Float, nullable=True)
    pe_wall_oi = Column(Float, nullable=True)
    buildup_pattern = Column(String(32), nullable=True)  # LONG_BUILD / SHORT_BUILD / SHORT_COVER / LONG_UNWIND
    gex_value = Column(Float, nullable=True)
    gex_interpretation = Column(String(64), nullable=True)
    chain_json = Column(JSON, nullable=True)  # Full chain blob for replay/analysis

    def __repr__(self):
        return f"<OISnapshot {self.index_name} {self.timestamp} spot={self.spot_price}>"


# ══════════════════════════════════════════════════════════════
# GLOBAL/SHARED TABLES — No user_id (same data for everyone)
# ══════════════════════════════════════════════════════════════

class AstroDaily(Base):
    __tablename__ = "astro_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Planet degrees (Lahiri ayanamsha sidereal)
    jupiter_deg = Column(Float, nullable=True)
    saturn_deg = Column(Float, nullable=True)
    mars_deg = Column(Float, nullable=True)
    venus_deg = Column(Float, nullable=True)
    mercury_deg = Column(Float, nullable=True)
    sun_deg = Column(Float, nullable=True)
    moon_deg = Column(Float, nullable=True)
    rahu_deg = Column(Float, nullable=True)
    ketu_deg = Column(Float, nullable=True)

    # Retrograde flags
    jupiter_retro = Column(Boolean, default=False)
    saturn_retro = Column(Boolean, default=False)
    mars_retro = Column(Boolean, default=False)
    venus_retro = Column(Boolean, default=False)
    mercury_retro = Column(Boolean, default=False)

    # Nakshatra
    nakshatra = Column(String(32), nullable=True)
    nakshatra_pada = Column(Integer, nullable=True)
    nakshatra_nature = Column(String(20), nullable=True)  # BULLISH / BEARISH / NEUTRAL

    # Tithi
    tithi = Column(Integer, nullable=True)
    tithi_name = Column(String(32), nullable=True)
    tithi_nature = Column(String(20), nullable=True)

    # Paksha
    paksha = Column(String(16), nullable=True)  # SHUKLA / KRISHNA

    # Hora & Yoga
    hora_sequence_json = Column(JSON, nullable=True)
    current_yoga = Column(String(32), nullable=True)

    # Aspects
    active_aspects_json = Column(JSON, nullable=True)

    # Composite
    astro_score = Column(Float, nullable=True)
    market_bias = Column(String(16), nullable=True)  # BULLISH / BEARISH / NEUTRAL

    def __repr__(self):
        return f"<AstroDaily {self.date} score={self.astro_score} bias={self.market_bias}>"


class WeeklyProbability(Base):
    __tablename__ = "weekly_probabilities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_start_date = Column(Date, nullable=False, index=True)
    day_of_week = Column(String(12), nullable=False)  # MONDAY..FRIDAY
    date = Column(Date, nullable=False, index=True)
    call_probability = Column(Float, nullable=True)
    put_probability = Column(Float, nullable=True)
    neutral_probability = Column(Float, nullable=True)
    predicted_bias = Column(String(16), nullable=True)  # BULLISH / BEARISH / NEUTRAL
    confidence = Column(Float, nullable=True)
    astro_events_json = Column(JSON, nullable=True)
    model_version = Column(String(16), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    actual_result = Column(String(16), nullable=True)  # BULLISH / BEARISH / NEUTRAL (filled post-market)
    prediction_correct = Column(Boolean, nullable=True)

    def __repr__(self):
        return f"<WeeklyProbability {self.date} bias={self.predicted_bias} conf={self.confidence}>"


class PatternBacktest(Base):
    __tablename__ = "pattern_backtests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_name = Column(String(64), nullable=False, index=True)
    pattern_type = Column(String(32), nullable=True)  # NAKSHATRA / TITHI / YOGA / ASPECT / COMPOSITE
    occurrences_total = Column(Integer, nullable=True)
    bull_count = Column(Integer, nullable=True)
    bear_count = Column(Integer, nullable=True)
    neutral_count = Column(Integer, nullable=True)
    win_rate_bull = Column(Float, nullable=True)
    win_rate_bear = Column(Float, nullable=True)
    avg_return_bull = Column(Float, nullable=True)
    avg_return_bear = Column(Float, nullable=True)
    best_strategy = Column(String(16), nullable=True)  # CE / PE / NEUTRAL
    data_from = Column(Date, nullable=True)
    data_till = Column(Date, nullable=True)
    last_updated = Column(DateTime, nullable=True, default=datetime.utcnow)

    def __repr__(self):
        return f"<PatternBacktest {self.pattern_name} wr_bull={self.win_rate_bull} wr_bear={self.win_rate_bear}>"


class DailyExport(Base):
    __tablename__ = "daily_exports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    export_type = Column(String(32), nullable=False)  # TRADES / OI / ASTRO / SECTOR / FULL
    file_path = Column(String(256), nullable=True)
    row_count = Column(Integer, nullable=True)
    status = Column(String(16), nullable=False, default="PENDING")  # PENDING / SUCCESS / FAILED
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return f"<DailyExport {self.date} {self.export_type} {self.status}>"


class SectorSnapshot(Base):
    __tablename__ = "sector_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    sector_name = Column(String(32), nullable=False)
    ltp = Column(Float, nullable=True)
    change_pct = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    impact_on_nifty = Column(Float, nullable=True)
    key_movers_json = Column(JSON, nullable=True)

    def __repr__(self):
        return f"<SectorSnapshot {self.sector_name} chg={self.change_pct}%>"
