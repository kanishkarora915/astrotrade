"""
═══════════════════════════════════════════════════════════════════════════════
ASTRONIFTY ENGINE — Main Entry Point (Multi-User)
AstroTrade by Kanishk Arora
═══════════════════════════════════════════════════════════════════════════════

Combines Vedic Astrology + OI Chain + FII/DII + Greeks + ML
for emotionless options trading on Nifty, BankNifty, GiftNifty.
Supports UNLIMITED users — each logs in with their own Zerodha API key.

Usage:
    python main.py                  # Start full engine + dashboard
    python main.py --dashboard-only # Only start dashboard server
    python main.py --weekly         # Generate weekly forecast only
    python main.py --backtest       # Run backtesting mode
"""

import sys
import os
import signal
import argparse
import threading
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

import config

# ═══════════════════════════════════════════════════════════════
# LOGGER SETUP
# ═══════════════════════════════════════════════════════════════
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/astronifty_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)


def print_banner():
    """Print startup banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║          ★  A S T R O N I F T Y   E N G I N E  ★            ║
    ║                                                               ║
    ║          AstroTrade by Kanishk Arora                         ║
    ║          MULTI-USER EDITION                                   ║
    ║                                                               ║
    ║   Vedic Astrology + OI Chain + FII/DII + Greeks + ML         ║
    ║   Emotionless Options Trading Engine                          ║
    ║                                                               ║
    ║   Indices: NIFTY | BANKNIFTY | GIFTNIFTY                     ║
    ║   + Sector Impact Analysis                                    ║
    ║   Unlimited Users — Each with their own Zerodha account       ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_dashboard_only():
    """Start only the dashboard server (users login via browser)."""
    port = int(os.getenv("PORT", "8888"))
    logger.info(f"Starting dashboard server only (multi-user) on port {port}...")
    import uvicorn
    from dashboard.app import app

    uvicorn.run(app, host="0.0.0.0", port=port)


def run_weekly_forecast():
    """Generate weekly forecast only (shared astro data)."""
    logger.info("Generating weekly probability forecast...")
    from analysis.probability import WeeklyProbabilityEngine
    from analysis.astro_engine import AstroEngine

    prob_engine = WeeklyProbabilityEngine()
    astro_engine = AstroEngine()

    spot = 23500  # Default
    iv = 0.15

    astro_forecast = astro_engine.get_weekly_astro_forecast(datetime.now())

    forecast = prob_engine.compute_next_week(
        spot=spot, iv=iv, astro_forecast=astro_forecast
    )

    logger.info("Weekly Forecast:")
    for day in forecast.get("daily_forecast", []):
        logger.info(
            f"  {day.get('day', 'N/A')} {day.get('date', 'N/A')}: "
            f"CE={day.get('call_probability', 0):.0%} PE={day.get('put_probability', 0):.0%} "
            f"Bias={day.get('bias', 'N/A')} | {day.get('nakshatra', 'N/A')}"
        )

    return forecast


def run_backtest():
    """Run backtesting mode."""
    logger.info("Starting backtest mode...")
    from database.db_manager import DBManager
    from ml.trainer import ModelTrainer

    db = DBManager()
    db.create_tables()

    trainer = ModelTrainer(db, None)
    results = trainer.backtest(
        strategy="astro_oi_combined",
        start_date=datetime(2024, 1, 1),
        end_date=datetime.now(),
    )

    logger.info("Backtest Results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")

    return results


def run_full_engine():
    """
    Start the full AstroNifty engine with multi-user support.

    Architecture:
    - Database initialized once (shared)
    - UserManager handles unlimited users (each with own Kite client)
    - MasterEngine runs shared analysis (astro, OI, price action)
    - Per-user: risk check, execution, position monitoring
    - Dashboard on port 8888 — users login with API key + secret
    - RealtimeHub: ONE WebSocket for market data, shared across all users
    """
    logger.info("Starting AstroNifty Engine — MULTI-USER MODE")
    logger.info(f"Indices: {', '.join(config.INDICES.keys())}")
    logger.info(f"Sectors: {', '.join(config.SECTORS.keys())}")
    logger.info(f"Default Paper Trade: {config.PAPER_TRADE}")

    # ── Initialize database ──
    from database.db_manager import DBManager

    db = DBManager()
    db.create_tables()
    logger.info("Database initialized (shared)")

    # ── Initialize UserManager ──
    from core.user_manager import UserManager

    user_manager = UserManager(db)
    user_manager.restore_sessions()
    active = user_manager.get_active_count()
    total = user_manager.get_user_count()
    logger.info(f"UserManager ready — {active} active / {total} total users")

    # ── Initialize core engine with user_manager ──
    from core.engine import MasterEngine

    engine = MasterEngine(user_manager=user_manager)
    logger.info("Master engine initialized (shared analysis + per-user execution)")

    # ── Setup signal handlers for graceful shutdown ──
    def graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(f"Received {sig_name} — initiating graceful shutdown...")
        logger.info("Closing all user positions...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    # ── Wire UserManager into dashboard module ──
    from dashboard.app import app, set_user_manager, set_engine
    set_user_manager(user_manager)
    set_engine(engine)

    # ── Start dashboard in background thread ──
    import uvicorn

    def start_dashboard():
        try:
            uvicorn.run(app, host="0.0.0.0", port=8888, log_level="warning")
        except Exception as e:
            logger.error(f"Dashboard failed: {e}")

    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    # Wait for dashboard to be ready
    import time
    for i in range(20):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:8888/health", timeout=2)
            logger.info("Dashboard responded on attempt {}", i + 1)
            break
        except Exception:
            pass

    logger.info("Dashboard LIVE at http://localhost:8888")
    logger.info("Login at http://localhost:8888/login")

    # ── Start the engine (scheduler + realtime hub) ──
    try:
        engine.start()

        # Keep main thread alive — engine runs via scheduler in background
        logger.info("Engine running. Press Ctrl+C to stop.")
        while engine.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt — shutting down...")
        engine.stop()
    except Exception as e:
        logger.critical(f"Engine crashed: {e}")
        engine.stop()
        raise


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="AstroNifty Engine — AstroTrade by Kanishk Arora (Multi-User)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                  Start full engine + dashboard (multi-user)
  python main.py --dashboard-only Start dashboard server only
  python main.py --weekly         Generate weekly forecast
  python main.py --backtest       Run backtesting

Multi-User:
  Users login at http://localhost:8888/login with their Zerodha API key.
  Each user gets isolated trades, positions, risk management.
  Shared: astro data, OI chain, price action, sector analysis.
        """,
    )

    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Start dashboard server only",
    )
    parser.add_argument(
        "--weekly", action="store_true", help="Generate weekly forecast only"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run backtesting mode"
    )

    args = parser.parse_args()

    # Create directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("exports", exist_ok=True)

    print_banner()

    if args.dashboard_only:
        run_dashboard_only()
    elif args.weekly:
        run_weekly_forecast()
    elif args.backtest:
        run_backtest()
    else:
        run_full_engine()


if __name__ == "__main__":
    main()
