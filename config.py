"""
AstroNifty Engine — Configuration
AstroTrade by Kanishk Arora
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# API CREDENTIALS
# ═══════════════════════════════════════════════════════════════
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", "")

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///astronifty.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

PAPER_TRADE = os.getenv("PAPER_TRADE", "True").lower() == "true"
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "500000"))


def get_current_expiry(index: str = "NIFTY") -> str:
    """Get the nearest expiry date for the given index.
    NIFTY: Thursday weekly, BANKNIFTY: Wednesday weekly.
    Returns date string YYYY-MM-DD.
    """
    from datetime import datetime, timedelta

    today = datetime.now().date()
    config = INDICES.get(index, INDICES.get("NIFTY", {}))
    expiry_day_name = config.get("expiry_day", "thursday")

    day_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
    }
    target_weekday = day_map.get(expiry_day_name, 3)

    # Find next occurrence of expiry day
    days_ahead = target_weekday - today.weekday()
    if days_ahead < 0:
        days_ahead += 7
    if days_ahead == 0:
        # Today is expiry day — use today if before 3:30 PM, else next week
        now = datetime.now()
        if now.hour >= 16:
            days_ahead = 7

    expiry_date = today + timedelta(days=days_ahead)
    return expiry_date.strftime("%Y-%m-%d")

# ═══════════════════════════════════════════════════════════════
# INDEX CONFIGURATION
# ═══════════════════════════════════════════════════════════════
INDICES = {
    "NIFTY": {
        "symbol": "NSE:NIFTY 50",
        "fut_symbol": "NFO:NIFTY{expiry}FUT",
        "lot_size": 65,
        "strike_gap": 50,
        "expiry_day": "thursday",
        "atm_range": 10,
    },
    "BANKNIFTY": {
        "symbol": "NSE:NIFTY BANK",
        "fut_symbol": "NFO:BANKNIFTY{expiry}FUT",
        "lot_size": 30,
        "strike_gap": 100,
        "expiry_day": "wednesday",
        "atm_range": 10,
    },
    "GIFTNIFTY": {
        "symbol": "NSE:NIFTY 50",  # Gift Nifty uses SGX data, mapped to Nifty
        "fut_symbol": "NFO:NIFTY{expiry}FUT",
        "lot_size": 65,
        "strike_gap": 50,
        "expiry_day": "thursday",
        "atm_range": 10,
        "is_gift": True,
    },
}

# ═══════════════════════════════════════════════════════════════
# SECTOR CONFIGURATION — For sector impact analysis
# ═══════════════════════════════════════════════════════════════
SECTORS = {
    "BANK": {
        "symbol": "NSE:NIFTY BANK",
        "weight_in_nifty": 0.33,
        "key_stocks": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
    },
    "IT": {
        "symbol": "NSE:NIFTY IT",
        "weight_in_nifty": 0.13,
        "key_stocks": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
    },
    "FINANCE": {
        "symbol": "NSE:NIFTY FIN SERVICE",
        "weight_in_nifty": 0.16,
        "key_stocks": ["BAJFINANCE", "BAJAJFINSV", "HDFCAMC", "SBILIFE"],
    },
    "AUTO": {
        "symbol": "NSE:NIFTY AUTO",
        "weight_in_nifty": 0.05,
        "key_stocks": ["MARUTI", "TATAMOTORS", "M&M", "HEROMOTOCO"],
    },
    "PHARMA": {
        "symbol": "NSE:NIFTY PHARMA",
        "weight_in_nifty": 0.04,
        "key_stocks": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
    },
    "METAL": {
        "symbol": "NSE:NIFTY METAL",
        "weight_in_nifty": 0.03,
        "key_stocks": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL"],
    },
    "ENERGY": {
        "symbol": "NSE:NIFTY ENERGY",
        "weight_in_nifty": 0.12,
        "key_stocks": ["RELIANCE", "ONGC", "BPCL", "IOC", "NTPC"],
    },
    "FMCG": {
        "symbol": "NSE:NIFTY FMCG",
        "weight_in_nifty": 0.08,
        "key_stocks": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA"],
    },
    "REALTY": {
        "symbol": "NSE:NIFTY REALTY",
        "weight_in_nifty": 0.01,
        "key_stocks": ["DLF", "GODREJPROP", "OBEROIRLTY"],
    },
    "MEDIA": {
        "symbol": "NSE:NIFTY MEDIA",
        "weight_in_nifty": 0.01,
        "key_stocks": ["ZEEL", "PVR"],
    },
}

# ═══════════════════════════════════════════════════════════════
# 100-POINT SCORING WEIGHTS
# ═══════════════════════════════════════════════════════════════
SCORING_WEIGHTS = {
    "oi_chain":      25,
    "oi_buildup":    22,
    "astro":         20,
    "greeks":        10,
    "price_action":  10,
    "fii_dii":        8,
    "global_cues":    7,
    "smart_money":    5,
    "expiry":         3,
    "breadth":        2,
}

# ═══════════════════════════════════════════════════════════════
# SIGNAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════
SIGNAL_THRESHOLDS = {
    "strong_bull":   85,
    "mild_bull":     70,
    "neutral_min":   45,
    "neutral_max":   55,
    "mild_bear":     30,
    "strong_bear":   15,
}

# ═══════════════════════════════════════════════════════════════
# RISK RULES
# ═══════════════════════════════════════════════════════════════
RISK_RULES = {
    "max_capital_per_trade": 0.20,
    "max_capital_per_index": 0.40,
    "daily_loss_limit":      0.03,
    "max_trades_per_day":    5,
    "no_entry_after":        "15:25",
    "signal_validity_mins":  120,
    "max_lots_first_3mo":    5,
    "consecutive_loss_pause": 3,
    "pause_duration_mins":   90,
    "vix_reduce_above":      18,
    "vix_block_above":       20,
}

# ═══════════════════════════════════════════════════════════════
# TIMING
# ═══════════════════════════════════════════════════════════════
TIMEZONE = "Asia/Kolkata"
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
PRE_MARKET_START = "09:00"
ENGINE_INTERVAL_SECONDS = 60
GIFT_NIFTY_OPEN = "06:30"
GIFT_NIFTY_CLOSE = "23:30"

# ═══════════════════════════════════════════════════════════════
# API RATE LIMITING
# ═══════════════════════════════════════════════════════════════
KITE_MAX_REQUESTS_PER_SEC = 3
KITE_RETRY_COUNT = 3
KITE_RETRY_DELAY_SEC = 2
