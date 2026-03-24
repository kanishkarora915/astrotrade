"""
AstroNifty Dashboard — FastAPI Application (Multi-User)
Main web server with JWT-based per-user authentication.
Each user logs in with their own Zerodha API key/secret and gets an isolated dashboard.
Shared data (astro, OI, sectors, prices) is the same for all users.
"""

import asyncio
import json
import os
import secrets
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import jwt
import pytz
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel

from core.user_manager import UserManager, UserSession
from database.db_manager import DBManager
from dashboard.websocket import DashboardWebSocket

IST = pytz.timezone("Asia/Kolkata")

# ---------------------------------------------------------------------------
# JWT Configuration
# ---------------------------------------------------------------------------
JWT_SECRET = os.environ.get("ASTRONIFTY_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24
JWT_COOKIE_NAME = "astronifty_token"

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------
logger.add(
    "logs/dashboard_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AstroNifty Engine",
    description="AstroTrade by Kanishk Arora — Astro + OI + Sector Trading Engine (Multi-User)",
    version="3.0.0",
)

ws_manager = DashboardWebSocket()

# Database and UserManager (initialized on startup)
db_manager: Optional[DBManager] = None
user_manager: Optional[UserManager] = None

# Shared data stores (same for all users — populated by engine modules)
_astro_data: dict = {}
_oi_data: dict = {}          # keyed by index name
_sector_data: dict = {}
_weekly_forecast: dict = {}
_live_prices: dict = {}       # real-time prices from RealtimeHub
_live_greeks: dict = {}       # real-time greeks from RealtimeHub
_module_health: dict = {
    "astro": "red",
    "oi": "red",
    "sector": "red",
    "scorer": "red",
    "risk": "red",
    "executor": "red",
    "database": "red",
    "telegram": "red",
    "ticker": "red",
}

# Reference to the engine's RealtimeHub (set at startup by engine)
_realtime_hub = None

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---------------------------------------------------------------------------
# JWT Helpers
# ---------------------------------------------------------------------------
def create_jwt(user_id: str, api_key: str) -> str:
    """Create a JWT token containing user_id and api_key."""
    payload = {
        "user_id": user_id,
        "api_key_masked": f"{api_key[:4]}****{api_key[-2:]}",
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Optional[dict]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("JWT expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug("Invalid JWT: {}", e)
        return None


def get_current_user(request: Request) -> UserSession:
    """Extract JWT from cookie, validate, and return the UserSession.
    Raises HTTPException 401 if invalid."""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please login at /login.",
        )

    payload = decode_jwt(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Session expired. Please login again.",
        )

    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token.")

    session = user_manager.get_session(user_id)
    if not session:
        raise HTTPException(
            status_code=401,
            detail="Session not found. Please login again.",
        )

    return session


def get_user_id_from_jwt(request: Request) -> Optional[str]:
    """Extract user_id from JWT cookie. Returns None if invalid (no exception)."""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        return None
    payload = decode_jwt(token)
    if not payload:
        return None
    return payload.get("user_id")


# ---------------------------------------------------------------------------
# Pydantic models for request bodies
# ---------------------------------------------------------------------------
class CredentialsInput(BaseModel):
    api_key: str
    api_secret: str


class ModeInput(BaseModel):
    mode: str  # "paper" or "live"


# ---------------------------------------------------------------------------
# Auth middleware — redirect unauthenticated requests to /login
# ---------------------------------------------------------------------------
AUTH_EXEMPT_PATHS = {
    "/login",
    "/api/auth/save-credentials",
    "/api/auth/callback",
    "/api/auth/status",
    "/api/admin/users",
    "/api/astro/live",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check JWT cookie for authentication. Exempt paths pass through."""
    path = request.url.path

    # Allow exempt paths
    if path in AUTH_EXEMPT_PATHS or path.startswith("/api/auth/"):
        return await call_next(request)

    # Allow static files
    if path.startswith("/static/"):
        return await call_next(request)

    # Check JWT authentication
    token = request.cookies.get(JWT_COOKIE_NAME)
    if token:
        payload = decode_jwt(token)
        if payload:
            user_id = payload.get("user_id")
            session = user_manager.get_session(user_id) if user_id else None
            if session and session.is_authenticated:
                return await call_next(request)

    # Not authenticated
    if path.startswith("/api/"):
        return JSONResponse(
            status_code=401,
            content={
                "error": "Not authenticated",
                "message": "Please login at /login first",
                "redirect": "/login",
            },
        )

    return RedirectResponse(url="/login", status_code=302)


# ---------------------------------------------------------------------------
# Helpers to update SHARED state (called by engine core)
# ---------------------------------------------------------------------------
def update_astro(data: dict):
    global _astro_data
    _astro_data = data


def update_oi(index: str, data: dict):
    global _oi_data
    _oi_data[index] = data


def update_sectors(data: dict):
    global _sector_data
    _sector_data = data


def update_weekly(data: dict):
    global _weekly_forecast
    _weekly_forecast = data


def set_module_health(module: str, status: str):
    """status must be 'green', 'yellow', or 'red'."""
    if module in _module_health and status in ("green", "yellow", "red"):
        _module_health[module] = status


def update_live_prices(data: dict):
    global _live_prices
    _live_prices = data


def update_live_greeks(data: dict):
    global _live_greeks
    _live_greeks = data


def set_realtime_hub(hub):
    """Set reference to the RealtimeHub for live data access."""
    global _realtime_hub
    _realtime_hub = hub
    logger.info("RealtimeHub reference set in dashboard app")


# ---------------------------------------------------------------------------
# Per-user data update helpers (called by engine with user_id)
# ---------------------------------------------------------------------------
def update_scores(data: dict, user_id: str = ""):
    """Update scores. If user_id given, update that user's session."""
    if user_id:
        user_manager.update_user_scores(user_id, data)
    else:
        # Update all authenticated users (legacy compat / shared scores)
        for sess in user_manager.get_all_authenticated_sessions():
            sess.scores = data


def update_signals(data: list, user_id: str = ""):
    if user_id:
        user_manager.update_user_signals(user_id, data)


def update_positions(data: list, user_id: str = ""):
    if user_id:
        user_manager.update_user_positions(user_id, data)


def update_trades(data: list, user_id: str = ""):
    if user_id:
        user_manager.update_user_trades(user_id, data)


def update_pnl(data: dict, user_id: str = ""):
    if user_id:
        user_manager.update_user_pnl(user_id, data)


# ---------------------------------------------------------------------------
# Routes — Authentication
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page."""
    login_path = TEMPLATES_DIR / "login.html"
    if not login_path.exists():
        raise HTTPException(status_code=500, detail="Login template not found")
    return HTMLResponse(content=login_path.read_text(encoding="utf-8"))


@app.post("/api/auth/save-credentials")
async def save_credentials(creds: CredentialsInput):
    """Register user via UserManager and return Kite login URL."""
    api_key = creds.api_key.strip()
    api_secret = creds.api_secret.strip()

    if not api_key or not api_secret:
        raise HTTPException(status_code=400, detail="API key and secret are required")

    if len(api_key) < 6:
        raise HTTPException(status_code=400, detail="Invalid API key format")

    # Register or login user (creates in DB if new, returns existing if found)
    try:
        session = user_manager.register_or_login(api_key, api_secret)
    except Exception as exc:
        logger.error("Failed to register user: {}", exc)
        raise HTTPException(status_code=500, detail=f"Failed to register: {exc}")

    logger.info(
        "Credentials saved for user {}.. (API key: {}...{})",
        session.user_id[:8], api_key[:4], api_key[-2:],
    )

    # If already authenticated, just return login URL for re-auth
    # Generate login URL
    try:
        login_url = session.get_login_url()
    except Exception as exc:
        logger.error("Failed to generate login URL: {}", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate login URL: {exc}")

    return JSONResponse(content={
        "success": True,
        "message": "Credentials saved. Redirecting to Zerodha...",
        "user_id": session.user_id,
        "api_key_masked": f"{api_key[:4]}****{api_key[-2:]}",
        "login_url": login_url,
    })


@app.get("/api/auth/callback")
async def auth_callback(request_token: str = Query(None), status: str = Query(None)):
    """Handle Kite redirect callback with request_token.

    Zerodha redirects here: /api/auth/callback?request_token=XXX&status=success

    The challenge: Kite callback doesn't tell us WHICH user is completing login.
    We try all unauthenticated sessions first, then all sessions as fallback.
    """
    if status != "success" or not request_token:
        logger.warning("Auth callback failed: status={}, token={}", status, request_token)
        return RedirectResponse(url="/login?error=auth_failed", status_code=302)

    # Try all sessions to find whose request_token this belongs to
    # Priority: unauthenticated users first (they just submitted credentials)
    authenticated_user_id = None
    auth_result = None

    # Get all sessions and try unauthenticated first
    all_sessions = []
    with user_manager._lock:
        for uid, sess in user_manager.sessions.items():
            all_sessions.append((uid, sess.is_authenticated))

    # Sort: unauthenticated first
    all_sessions.sort(key=lambda x: x[1])

    for uid, _ in all_sessions:
        result = user_manager.authenticate_user(uid, request_token)
        if result.get("success"):
            authenticated_user_id = uid
            auth_result = result
            break

    if not authenticated_user_id:
        logger.error("Auth callback: no user matched for request_token")
        return RedirectResponse(url="/login?error=no_matching_user", status_code=302)

    session = user_manager.get_session(authenticated_user_id)
    logger.success(
        "Auth callback successful. User: {} ({}) — user_id: {}",
        auth_result.get("zerodha_user_name", ""),
        auth_result.get("zerodha_user_id", ""),
        authenticated_user_id[:8],
    )

    # Create JWT and set cookie
    token = create_jwt(authenticated_user_id, session.api_key)
    response = RedirectResponse(url="/?auth=success", status_code=302)
    response.set_cookie(
        key=JWT_COOKIE_NAME,
        value=token,
        httponly=False,
        samesite="lax",
        max_age=JWT_EXPIRY_HOURS * 3600,
        path="/",
    )

    # Start background data fetching for this user's Kite session
    asyncio.create_task(_start_live_data(authenticated_user_id))

    return response


@app.get("/api/auth/status")
async def auth_status(request: Request):
    """Check current authentication status from JWT cookie."""
    token = request.cookies.get(JWT_COOKIE_NAME)
    if not token:
        return JSONResponse(content={
            "authenticated": False,
            "message": "No session found",
        })

    payload = decode_jwt(token)
    if not payload:
        return JSONResponse(content={
            "authenticated": False,
            "message": "Session expired",
        })

    user_id = payload.get("user_id")
    session = user_manager.get_session(user_id) if user_id else None

    if not session:
        return JSONResponse(content={
            "authenticated": False,
            "message": "Session not found",
        })

    return JSONResponse(content=session.get_auth_status())


@app.post("/api/auth/logout")
async def logout(request: Request):
    """Clear JWT cookie and log out."""
    user_id = get_user_id_from_jwt(request)
    if user_id:
        session = user_manager.get_session(user_id)
        if session:
            session.invalidate()
        logger.info("User {} logged out", user_id[:8])

    response = JSONResponse(content={"success": True, "message": "Logged out"})
    response.delete_cookie(key=JWT_COOKIE_NAME, path="/")
    return response


@app.post("/api/auth/mode")
async def set_mode(request: Request, mode_input: ModeInput):
    """Set paper/live mode for current user."""
    session = get_current_user(request)
    if mode_input.mode not in ("paper", "live"):
        raise HTTPException(status_code=400, detail="Mode must be 'paper' or 'live'")
    session.paper_trade = (mode_input.mode == "paper")
    logger.info("User {} switched to {} mode", session.user_id[:8], mode_input.mode)
    return JSONResponse(content={"success": True, "mode": mode_input.mode})


# ---------------------------------------------------------------------------
# Routes — Public Astro (no auth needed)
# ---------------------------------------------------------------------------
@app.get("/api/astro/live")
async def astro_live():
    """Return current astro data (public — no auth needed)."""
    if _astro_data:
        return JSONResponse(content=_astro_data)
    # Compute fresh if not available
    try:
        from data.astro_feed import AstroFeed
        from analysis.astro_engine import AstroEngine
        feed = AstroFeed()
        snapshot = feed.get_current_snapshot()
        engine = AstroEngine()
        result = engine.score(snapshot)
        return JSONResponse(content=result)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc), "traceback": True}, status_code=500)


# ---------------------------------------------------------------------------
# Routes — Dashboard
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard template not found")
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Routes — Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """Return health status of every engine module."""
    all_green = all(v == "green" for v in _module_health.values())
    any_red = any(v == "red" for v in _module_health.values())
    overall = "green" if all_green else ("red" if any_red else "yellow")
    return JSONResponse(
        content={
            "status": overall,
            "modules": _module_health,
            "timestamp": datetime.now().isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Routes — Admin
# ---------------------------------------------------------------------------
@app.get("/api/admin/users")
async def admin_users():
    """Return total and active user count (no sensitive data)."""
    stats = user_manager.get_stats()
    stats["ws_connections"] = ws_manager.connection_count
    stats["ws_online_users"] = ws_manager.online_user_count
    return JSONResponse(content=stats)


# ---------------------------------------------------------------------------
# Routes — User-scoped API
# ---------------------------------------------------------------------------
@app.get("/api/scores")
async def get_scores(request: Request):
    """Current composite scores for the logged-in user."""
    session = get_current_user(request)
    return JSONResponse(content={
        "scores": session.scores,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/signals")
async def get_signals(request: Request):
    """Active trading signals for the logged-in user."""
    session = get_current_user(request)
    return JSONResponse(content={
        "signals": session.signals,
        "count": len(session.signals),
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/positions")
async def get_positions(request: Request):
    """Open positions with live P&L for the logged-in user."""
    session = get_current_user(request)
    total_pnl = sum(p.get("pnl", 0) for p in session.positions)
    return JSONResponse(content={
        "positions": session.positions,
        "count": len(session.positions),
        "total_pnl": round(total_pnl, 2),
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/trades/today")
async def get_trades_today(request: Request):
    """Today's trade log for the logged-in user."""
    session = get_current_user(request)
    return JSONResponse(content={
        "trades": session.trades_today,
        "count": len(session.trades_today),
        "date": date.today().isoformat(),
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/pnl")
async def get_pnl(request: Request):
    """Today's P&L summary for the logged-in user."""
    session = get_current_user(request)
    return JSONResponse(content={
        "pnl": session.pnl_summary,
        "date": date.today().isoformat(),
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# Routes — Shared API (same for all users)
# ---------------------------------------------------------------------------
@app.get("/api/astro")
async def get_astro():
    """Current astro data — shared across all users."""
    return JSONResponse(content={
        "astro": _astro_data,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/oi/{index}")
async def get_oi(index: str):
    """OI chain for a specific index — shared data."""
    index_upper = index.upper()
    if index_upper not in _oi_data:
        raise HTTPException(
            status_code=404,
            detail=f"No OI data for index '{index_upper}'. Available: {list(_oi_data.keys())}",
        )
    return JSONResponse(content={
        "index": index_upper,
        "oi": _oi_data[index_upper],
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/sectors")
async def get_sectors():
    """Sector impact / heatmap data — shared."""
    return JSONResponse(content={
        "sectors": _sector_data,
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/api/weekly")
async def get_weekly():
    """Weekly probability forecast — shared."""
    return JSONResponse(content={
        "forecast": _weekly_forecast,
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# Routes — Real-time API (from RealtimeHub) — Shared
# ---------------------------------------------------------------------------
@app.get("/api/live/prices")
async def get_live_prices():
    """Real-time prices from RealtimeHub tick cache."""
    if _realtime_hub:
        data = {
            idx: {"ltp": _realtime_hub.get_live_price_by_index(idx)}
            for idx in _realtime_hub._spot_tokens
        }
    else:
        data = _live_prices
    return JSONResponse(content={"prices": data, "timestamp": datetime.now().isoformat()})


@app.get("/api/live/chain/{index}")
async def get_live_chain(index: str):
    """Real-time option chain from RealtimeHub."""
    index_upper = index.upper()
    if _realtime_hub:
        chain = _realtime_hub.get_live_chain(index_upper)
        if not chain.empty:
            return JSONResponse(content={
                "index": index_upper,
                "chain": chain.to_dict(orient="records"),
                "timestamp": datetime.now().isoformat(),
            })
    raise HTTPException(status_code=404, detail=f"No live chain for '{index_upper}'")


@app.get("/api/live/greeks/{index}")
async def get_live_greeks(index: str):
    """Real-time greeks from RealtimeHub."""
    index_upper = index.upper()
    if _realtime_hub:
        greeks = _realtime_hub.get_live_greeks(index_upper)
        if greeks:
            return JSONResponse(content={
                "index": index_upper,
                "greeks": greeks,
                "timestamp": datetime.now().isoformat(),
            })
    raise HTTPException(status_code=404, detail=f"No live greeks for '{index_upper}'")


@app.get("/api/live/status")
async def get_live_status():
    """RealtimeHub connection and tick status."""
    if _realtime_hub:
        return JSONResponse(content=_realtime_hub.health_check())
    return JSONResponse(content={"status": "not_initialized"})


# ---------------------------------------------------------------------------
# Routes — PDF Export (per-user)
# ---------------------------------------------------------------------------
EXPORTS_DIR = Path(__file__).parent.parent.parent / "exports"


@app.get("/api/export/pdf")
async def export_pdf_today(request: Request):
    """Generate and download today's PDF report for the current user."""
    session = get_current_user(request)
    today_str = date.today().strftime("%Y%m%d")

    # Per-user export directory
    user_export_dir = EXPORTS_DIR / session.user_id[:8]
    user_export_dir.mkdir(parents=True, exist_ok=True)
    pdf_filename = f"ASTRONIFTY_{session.zerodha_user_id or session.user_id[:8]}_{today_str}.pdf"
    pdf_path = user_export_dir / pdf_filename

    if pdf_path.exists():
        return FileResponse(
            path=str(pdf_path),
            filename=pdf_filename,
            media_type="application/pdf",
        )

    try:
        from database.pdf_exporter import DailyPDFExporter

        pdf_exporter = DailyPDFExporter(exports_dir=str(user_export_dir))
        engine_data = {
            "scores": session.scores,
            "oi_data": _oi_data,
            "astro_data": _astro_data,
            "signals": session.signals,
            "trades": session.trades_today,
            "pnl_summary": session.pnl_summary,
            "sector_data": _sector_data,
            "weekly_forecast": _weekly_forecast,
            "user_name": session.zerodha_user_name,
            "user_id": session.zerodha_user_id,
        }
        generated_path = pdf_exporter.generate_daily_report(
            report_date=date.today(),
            engine_data=engine_data,
        )
        return FileResponse(
            path=generated_path,
            filename=pdf_filename,
            media_type="application/pdf",
        )
    except Exception as e:
        logger.error("PDF generation failed for user {}: {}", session.user_id[:8], str(e))
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# WebSocket — Per-user live data stream
# ---------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint. First message must be 'auth' with JWT token."""
    await ws_manager.connect(websocket, user_id="")
    logger.info("WebSocket client connected (awaiting auth): {}", websocket.client)

    user_id = None
    session = None

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                msg_type = msg.get("type", "")

                # ── Authentication (first message should be auth) ──
                if msg_type == "auth":
                    token = msg.get("token", "")
                    payload = decode_jwt(token)
                    if payload:
                        user_id = payload.get("user_id")
                        session = user_manager.get_session(user_id) if user_id else None

                    if not session or not session.is_authenticated:
                        await ws_manager.send_personal(websocket, {
                            "type": "auth_error",
                            "message": "Invalid or expired token. Please re-login.",
                        })
                        break

                    # Associate this connection with the user
                    ws_manager.set_user(websocket, user_id)
                    logger.info("WebSocket authenticated: user={}", user_id[:8])

                    # Send initial snapshot with user-specific + shared data
                    snapshot = {
                        "type": "snapshot",
                        "channel": "snapshot",
                        "user": {
                            "user_id": session.zerodha_user_id,
                            "user_name": session.zerodha_user_name,
                            "paper_trade": session.paper_trade,
                            "capital": session.capital,
                        },
                        "scores": session.scores,
                        "signals": session.signals,
                        "positions": session.positions,
                        "astro": _astro_data,
                        "oi": _oi_data,
                        "sectors": _sector_data,
                        "weekly": _weekly_forecast,
                        "trades": session.trades_today,
                        "pnl": session.pnl_summary,
                        "health": _module_health,
                        "timestamp": datetime.now().isoformat(),
                    }

                    if _realtime_hub:
                        live_data = _realtime_hub.get_all_live_data()
                        snapshot["live_prices"] = live_data.get("prices", {})
                        snapshot["live_chains"] = live_data.get("chains", {})
                        snapshot["live_greeks"] = live_data.get("greeks", {})
                        snapshot["live_sectors"] = live_data.get("sectors", {})
                        snapshot["tick_count"] = live_data.get("tick_count", 0)
                        snapshot["ticker_connected"] = live_data.get("connected", False)
                        snapshot["subscribed_count"] = live_data.get("subscribed_count", 0)

                    await ws_manager.send_personal(websocket, snapshot)
                    await ws_manager.send_personal(websocket, {
                        "type": "auth_success",
                        "message": "Authenticated",
                    })

                elif msg_type == "ping":
                    await ws_manager.send_personal(websocket, {"type": "pong", "channel": "pong"})

                elif msg_type == "subscribe":
                    channels = msg.get("channels", [])
                    ws_manager.subscribe_client(websocket, channels)
                    await ws_manager.send_personal(websocket, {
                        "type": "subscribed",
                        "channel": "system",
                        "channels": channels,
                    })

                elif msg_type == "unsubscribe":
                    channels = msg.get("channels", [])
                    ws_manager.unsubscribe_client(websocket, channels)
                    await ws_manager.send_personal(websocket, {
                        "type": "unsubscribed",
                        "channel": "system",
                        "channels": channels,
                    })

                elif msg_type == "get_channels":
                    await ws_manager.send_personal(websocket, {
                        "type": "channels_list",
                        "channel": "system",
                        "channels": sorted(ws_manager._client_channels.get(websocket, set())),
                    })

                else:
                    logger.debug("Unknown WS message type: {}", msg_type)

            except json.JSONDecodeError:
                logger.warning("Invalid JSON from WS client")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info("WebSocket disconnected: {} (user={})", websocket.client, user_id[:8] if user_id else "anon")
    except Exception as e:
        ws_manager.disconnect(websocket)
        logger.error("WebSocket error: {} (user={})", str(e), user_id[:8] if user_id else "anon")


# ---------------------------------------------------------------------------
# Broadcast helpers — Channel-based (called by engine / RealtimeHub)
# ---------------------------------------------------------------------------

# ── SHARED broadcasts (go to ALL users) ──

async def broadcast_price_update(prices: dict):
    """Push live prices to all users (every second from RealtimeHub)."""
    update_live_prices(prices)
    await ws_manager.broadcast_shared({
        "channel": "prices", "type": "prices", "data": prices,
        "ts": datetime.now().isoformat(),
    })


async def broadcast_chain_update(index: str, chain_data: list):
    """Push rebuilt OI chain for an index to all users."""
    update_oi(index, {"chain": chain_data})
    await ws_manager.broadcast_shared({
        "channel": "chain", "type": "chain", "index": index, "data": chain_data,
        "ts": datetime.now().isoformat(),
    })


async def broadcast_greeks_update(index: str, greeks: dict):
    """Push real-time greeks to all users."""
    update_live_greeks({index: greeks})
    await ws_manager.broadcast_shared({
        "channel": "greeks", "type": "greeks", "index": index, "data": greeks,
        "ts": datetime.now().isoformat(),
    })


async def broadcast_sector_update(sectors: dict):
    """Push sector data to all users."""
    update_sectors(sectors)
    await ws_manager.broadcast_shared({
        "channel": "sectors", "type": "sectors", "data": sectors,
        "ts": datetime.now().isoformat(),
    })


async def broadcast_astro_update(astro: dict):
    """Push astro data to all users."""
    update_astro(astro)
    await ws_manager.broadcast_shared({
        "channel": "astro", "type": "astro", "data": astro,
        "ts": datetime.now().isoformat(),
    })


async def broadcast_health_update():
    """Push module health status to all users."""
    ticker_status = "red"
    if _realtime_hub:
        hc = _realtime_hub.health_check()
        ticker_status = hc.get("status", "red")
    _module_health["ticker"] = ticker_status
    await ws_manager.broadcast_shared({
        "channel": "health", "type": "health", "data": _module_health,
        "ts": datetime.now().isoformat(),
    })


# ── Legacy compat: OI update ──
async def broadcast_oi_update(index: str, oi: dict):
    update_oi(index, oi)
    await ws_manager.broadcast_shared({
        "channel": "chain", "type": "oi", "index": index, "data": oi,
        "ts": datetime.now().isoformat(),
    })


# ── PER-USER broadcasts ──

async def broadcast_score_update(scores: dict, user_id: str = ""):
    """Push score update to a specific user, or all users."""
    ts = datetime.now().isoformat()
    payload = {"channel": "scores", "type": "scores", "data": scores, "ts": ts}

    if user_id:
        user_manager.update_user_scores(user_id, scores)
        await ws_manager.broadcast_to_user(user_id, payload)
    else:
        for sess in user_manager.get_all_authenticated_sessions():
            sess.scores = scores
            await ws_manager.broadcast_to_user(sess.user_id, payload)


async def broadcast_signal(signal: dict, user_id: str = ""):
    """Push a new signal to a specific user."""
    ts = datetime.now().isoformat()
    payload = {"channel": "signals", "type": "signal", "data": signal, "ts": ts}

    if user_id:
        user_manager.append_user_signal(user_id, signal)
        await ws_manager.broadcast_to_user(user_id, payload)
    else:
        for sess in user_manager.get_all_authenticated_sessions():
            sess.signals.append(signal)
            await ws_manager.broadcast_to_user(sess.user_id, payload)


async def broadcast_position_update(positions: list, user_id: str = ""):
    """Push position update to a specific user."""
    total_pnl = sum(p.get("pnl", 0) for p in positions)
    ts = datetime.now().isoformat()
    payload = {
        "channel": "positions", "type": "positions", "data": positions,
        "total_pnl": round(total_pnl, 2), "ts": ts,
    }

    if user_id:
        user_manager.update_user_positions(user_id, positions)
        await ws_manager.broadcast_to_user(user_id, payload)
    else:
        for sess in user_manager.get_all_authenticated_sessions():
            sess.positions = positions
            await ws_manager.broadcast_to_user(sess.user_id, payload)


async def broadcast_trade(trade: dict, user_id: str = ""):
    """Push a trade to a specific user."""
    ts = datetime.now().isoformat()
    payload = {"channel": "trades", "type": "trade", "data": trade, "ts": ts}

    if user_id:
        user_manager.append_user_trade(user_id, trade)
        await ws_manager.broadcast_to_user(user_id, payload)
    else:
        for sess in user_manager.get_all_authenticated_sessions():
            sess.trades_today.append(trade)
            await ws_manager.broadcast_to_user(sess.user_id, payload)


async def broadcast_pnl_update(pnl: dict, user_id: str = ""):
    """Push P&L update to a specific user."""
    ts = datetime.now().isoformat()
    payload = {"channel": "pnl", "type": "pnl", "data": pnl, "ts": ts}

    if user_id:
        user_manager.update_user_pnl(user_id, pnl)
        await ws_manager.broadcast_to_user(user_id, payload)
    else:
        for sess in user_manager.get_all_authenticated_sessions():
            sess.pnl_summary = pnl
            await ws_manager.broadcast_to_user(sess.user_id, payload)


# ---------------------------------------------------------------------------
# Session auto-refresh scheduler (6:05 AM IST daily)
# ---------------------------------------------------------------------------
async def _session_refresh_loop():
    """Background task: at 6:05 AM IST daily, invalidate ALL user sessions
    and notify all dashboard clients."""
    while True:
        now = datetime.now(IST)
        target = now.replace(hour=6, minute=5, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)

        wait_seconds = (target - now).total_seconds()
        logger.info(
            "Session refresh scheduled for {} IST ({:.0f}s from now)",
            target.strftime("%Y-%m-%d %H:%M"),
            wait_seconds,
        )
        await asyncio.sleep(wait_seconds)

        # Invalidate ALL sessions
        logger.warning("6:05 AM IST — Invalidating ALL user sessions")
        user_manager.invalidate_all_sessions()

        # Broadcast session_expired to ALL connected WebSocket clients
        try:
            await ws_manager.broadcast_all({
                "type": "session_expired",
                "channel": "system",
                "message": "Kite session expired (6 AM daily reset). Please re-login.",
                "ts": datetime.now(IST).isoformat(),
            })
        except Exception as exc:
            logger.error("Failed to broadcast session expiry: {}", exc)


_live_data_running = False


async def _start_live_data(user_id: str):
    """Start live market data fetching using first authenticated user's Kite session."""
    global _live_data_running, _oi_data, _live_prices, _sector_data
    if _live_data_running:
        return  # Already running from another user

    session = user_manager.get_session(user_id)
    if not session or not session.kite_client:
        logger.warning("Cannot start live data — no Kite client for user {}", user_id[:8])
        return

    _live_data_running = True
    logger.info("Starting live data fetching using user {}'s Kite session", user_id[:8])

    kite = session.kite_client

    async def _fetch_loop():
        global _oi_data, _live_prices, _sector_data
        while _live_data_running:
            try:
                # Fetch live prices for indices
                import asyncio as _aio
                prices = {}
                try:
                    ltp_data = kite.get_ltp([
                        "NSE:NIFTY 50", "NSE:NIFTY BANK",
                        "NSE:NIFTY FIN SERVICE", "NSE:NIFTY IT",
                        "NSE:NIFTY AUTO", "NSE:NIFTY PHARMA",
                        "NSE:NIFTY METAL", "NSE:NIFTY ENERGY",
                        "NSE:NIFTY FMCG", "NSE:NIFTY REALTY",
                        "NSE:NIFTY MEDIA",
                    ])
                    for sym, data in ltp_data.items():
                        prices[sym] = data
                    _live_prices = prices
                    set_module_health("ticker", "green")
                except Exception as e:
                    logger.error("LTP fetch error: {}", e)
                    set_module_health("ticker", "red")

                # Fetch OI chain for NIFTY
                try:
                    from analysis.oi_chain import OIChainAnalyzer
                    oi = OIChainAnalyzer()
                    for index_name in ["NIFTY", "BANKNIFTY"]:
                        chain = kite.get_option_chain(index_name)
                        if chain is not None and not chain.empty:
                            spot = 0
                            if index_name == "NIFTY":
                                spot = ltp_data.get("NSE:NIFTY 50", {}).get("last_price", 0)
                            elif index_name == "BANKNIFTY":
                                spot = ltp_data.get("NSE:NIFTY BANK", {}).get("last_price", 0)
                            oi_result = oi.score(chain, None, spot)
                            _oi_data[index_name] = oi_result
                    set_module_health("oi", "green")
                except Exception as e:
                    logger.error("OI fetch error: {}", e)
                    set_module_health("oi", "red")

                # Broadcast to all connected users
                if prices:
                    await ws_manager.broadcast_shared({
                        "channel": "prices",
                        "data": prices,
                    })
                if _oi_data:
                    await ws_manager.broadcast_shared({
                        "channel": "oi",
                        "data": _oi_data,
                    })

            except Exception as exc:
                logger.error("Live data loop error: {}", exc)

            await asyncio.sleep(5)  # Every 5 seconds

    asyncio.create_task(_fetch_loop())
    logger.success("Live data fetch loop started")


async def _init_shared_data():
    """Compute shared data that doesn't need API keys (astro, weekly forecast)."""
    global _astro_data, _weekly_forecast
    try:
        import traceback as _tb
        from data.astro_feed import AstroFeed
        from analysis.astro_engine import AstroEngine
        logger.info("Computing astro data...")
        feed = AstroFeed()
        logger.info("AstroFeed created, getting snapshot...")
        snapshot = feed.get_current_snapshot()
        logger.info("Snapshot ready, computing score...")
        astro = AstroEngine()
        _astro_data = astro.score(snapshot)
        set_module_health("astro", "green")
        logger.success("Astro data computed on startup: score={}, bias={}",
                       _astro_data.get("score"), _astro_data.get("bias"))
    except Exception as exc:
        import traceback as _tb
        logger.error("Failed to compute astro data on startup: {}\n{}", exc, _tb.format_exc())
        set_module_health("astro", "red")

    try:
        from analysis.probability import WeeklyProbabilityEngine
        from analysis.astro_engine import AstroEngine as AE2
        prob = WeeklyProbabilityEngine()
        astro_eng = AE2()
        astro_fc = astro_eng.get_weekly_astro_forecast(datetime.now())
        _weekly_forecast = prob.compute_next_week(spot=23500, iv=0.15, astro_forecast=astro_fc)
        logger.success("Weekly forecast computed on startup")
    except Exception as exc:
        logger.error("Failed to compute weekly forecast: {}", exc)

    # Schedule periodic astro refresh every 30 minutes
    asyncio.create_task(_astro_refresh_loop())


async def _astro_refresh_loop():
    """Refresh astro data every 30 minutes and broadcast to all connected clients."""
    global _astro_data
    while True:
        await asyncio.sleep(1800)  # 30 min
        try:
            from data.astro_feed import AstroFeed
            from analysis.astro_engine import AstroEngine
            feed = AstroFeed()
            snapshot = feed.get_current_snapshot()
            astro = AstroEngine()
            _astro_data = astro.score(snapshot)
            set_module_health("astro", "green")
            # Broadcast to all connected users
            await ws_manager.broadcast_shared({
                "channel": "astro",
                "data": _astro_data,
            })
            logger.info("Astro refreshed: score={}, bias={}", _astro_data.get("score"), _astro_data.get("bias"))
        except Exception as exc:
            logger.error("Astro refresh failed: {}", exc)


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    global db_manager, user_manager

    logger.info("AstroNifty Dashboard (Multi-User) starting on http://0.0.0.0:8888")
    logger.info("JWT secret configured ({}...)", JWT_SECRET[:8])

    # Initialize database manager and create tables
    try:
        db_manager = DBManager()
        db_manager.create_tables()
        set_module_health("database", "green")
        logger.success("Database manager initialized + tables created")
    except Exception as exc:
        logger.error("Failed to initialize database: {}", exc)
        set_module_health("database", "red")
        raise

    # Initialize user manager with DB (only if not already injected by main.py)
    if user_manager is None:
        user_manager = UserManager(db_manager)

    # Restore persisted user sessions from DB
    user_manager.restore_sessions()
    stats = user_manager.get_stats()
    logger.info(
        "Restored {} users ({} authenticated)",
        stats["total_users"],
        stats["active_users"],
    )

    # Start the daily session refresh scheduler
    asyncio.create_task(_session_refresh_loop())
    logger.info("Session refresh scheduler started (6:05 AM IST daily)")

    # Auto-compute astro data on startup (no API key needed)
    asyncio.create_task(_init_shared_data())

    # Auto-start live data if any restored user has active Kite session
    if stats["active_users"] > 0:
        for uid, sess in user_manager.sessions.items():
            if sess.is_authenticated and sess.kite_client:
                asyncio.create_task(_start_live_data(uid))
                logger.info("Auto-started live data for restored user {}", uid[:8])
                break


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("AstroNifty Dashboard shutting down")
    for conn in list(ws_manager.active_connections):
        try:
            await conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# External wiring — called by main.py when running full engine
# ---------------------------------------------------------------------------
_engine_ref = None


def set_user_manager(um):
    """Inject a pre-built UserManager (from main.py) so dashboard shares it."""
    global user_manager
    if um is not None:
        user_manager = um
        logger.info("UserManager injected from main engine ({} users)", um.get_user_count())


def set_engine(engine):
    """Store reference to MasterEngine for dashboard to query live state."""
    global _engine_ref
    _engine_ref = engine
    logger.info("MasterEngine reference set in dashboard")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_dashboard(host: str = "0.0.0.0", port: int = 8888):
    """Start the dashboard server (blocking call)."""
    logger.info("Launching AstroNifty Dashboard — {}:{}", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
