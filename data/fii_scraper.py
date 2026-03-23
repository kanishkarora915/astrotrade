"""
FIIScraper - Fetches FII/DII cash and derivatives data from NSE India.

Handles NSE cookie/session requirements, parses daily FII/DII trade data,
and scores the institutional flow for bullish/bearish bias.
"""

import httpx
from datetime import datetime
from loguru import logger


class FIIScraper:
    """
    Scrapes FII/DII data from NSE India APIs.

    NSE requires a valid session cookie obtained by first hitting the main page
    before calling any API endpoint. This class manages that flow.
    """

    BASE_URL = "https://www.nseindia.com"
    FII_DII_URL = f"{BASE_URL}/api/fiidiiTradeReact"
    FII_DERIVATIVES_URL = f"{BASE_URL}/api/participant-wise-oi-data"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }

    TIMEOUT = 15.0

    def __init__(self):
        self._cookies: dict = {}
        self._cookie_timestamp: datetime | None = None
        self._cookie_max_age_seconds = 300  # refresh cookies every 5 min

    # ------------------------------------------------------------------
    # Cookie / session management
    # ------------------------------------------------------------------
    def _cookies_stale(self) -> bool:
        if not self._cookies or self._cookie_timestamp is None:
            return True
        elapsed = (datetime.now() - self._cookie_timestamp).total_seconds()
        return elapsed > self._cookie_max_age_seconds

    async def _refresh_cookies(self, client: httpx.AsyncClient) -> None:
        """Hit the NSE homepage to obtain valid session cookies."""
        try:
            resp = await client.get(
                self.BASE_URL,
                headers=self.HEADERS,
                follow_redirects=True,
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
            self._cookies = dict(resp.cookies)
            self._cookie_timestamp = datetime.now()
            logger.debug("NSE cookies refreshed: {} cookies obtained", len(self._cookies))
        except Exception as exc:
            logger.error("Failed to refresh NSE cookies: {}", exc)
            self._cookies = {}

    async def _get_client(self) -> httpx.AsyncClient:
        """Return an AsyncClient with valid cookies."""
        client = httpx.AsyncClient(
            headers=self.HEADERS,
            follow_redirects=True,
            timeout=self.TIMEOUT,
        )
        if self._cookies_stale():
            await self._refresh_cookies(client)
        return client

    async def _nse_get(self, url: str) -> dict | list | None:
        """Perform an authenticated GET against NSE API."""
        client = await self._get_client()
        try:
            resp = await client.get(url, cookies=self._cookies)
            if resp.status_code == 401 or resp.status_code == 403:
                logger.warning("NSE returned {}; refreshing cookies and retrying.", resp.status_code)
                await self._refresh_cookies(client)
                resp = await client.get(url, cookies=self._cookies)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.error("NSE GET {} failed: {}", url, exc)
            return None
        finally:
            await client.aclose()

    # ------------------------------------------------------------------
    # Sync wrappers (for non-async callers)
    # ------------------------------------------------------------------
    def _sync_nse_get(self, url: str) -> dict | list | None:
        """Synchronous wrapper using httpx (no async)."""
        try:
            with httpx.Client(
                headers=self.HEADERS,
                follow_redirects=True,
                timeout=self.TIMEOUT,
            ) as client:
                # Get cookies first
                home_resp = client.get(self.BASE_URL)
                home_resp.raise_for_status()
                cookies = dict(home_resp.cookies)

                resp = client.get(url, cookies=cookies)
                if resp.status_code in (401, 403):
                    logger.warning("NSE returned {}; retrying with fresh cookies.", resp.status_code)
                    home_resp = client.get(self.BASE_URL)
                    cookies = dict(home_resp.cookies)
                    resp = client.get(url, cookies=cookies)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("Sync NSE GET {} failed: {}", url, exc)
            return None

    # ------------------------------------------------------------------
    # 1. FII/DII Cash Market Data
    # ------------------------------------------------------------------
    async def fetch_fii_dii_data(self) -> dict:
        """
        Fetch daily FII/DII cash market buy/sell data from NSE.

        Returns
        -------
        dict with keys: fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net,
                        date, raw_data, source
        """
        try:
            data = await self._nse_get(self.FII_DII_URL)

            if not data:
                logger.warning("No FII/DII data returned from NSE.")
                return self._empty_fii_dii("No data from NSE")

            # NSE returns a list of category-wise entries
            fii_buy = 0.0
            fii_sell = 0.0
            dii_buy = 0.0
            dii_sell = 0.0
            trade_date = ""

            for entry in data if isinstance(data, list) else [data]:
                category = entry.get("category", "").upper()
                buy_val = self._parse_number(entry.get("buyValue", entry.get("BUY_VALUE", 0)))
                sell_val = self._parse_number(entry.get("sellValue", entry.get("SELL_VALUE", 0)))
                date_str = entry.get("date", entry.get("DATE1", ""))

                if "FII" in category or "FPI" in category:
                    fii_buy += buy_val
                    fii_sell += sell_val
                    if date_str:
                        trade_date = date_str
                elif "DII" in category:
                    dii_buy += buy_val
                    dii_sell += sell_val
                    if date_str and not trade_date:
                        trade_date = date_str

            result = {
                "fii_buy": round(fii_buy, 2),
                "fii_sell": round(fii_sell, 2),
                "fii_net": round(fii_buy - fii_sell, 2),
                "dii_buy": round(dii_buy, 2),
                "dii_sell": round(dii_sell, 2),
                "dii_net": round(dii_buy - dii_sell, 2),
                "date": trade_date,
                "source": "NSE",
                "raw_data": data,
            }

            logger.info(
                "FII/DII cash | date={} | FII net={:.2f} cr | DII net={:.2f} cr",
                trade_date, result["fii_net"], result["dii_net"],
            )
            return result

        except Exception as exc:
            logger.error("fetch_fii_dii_data failed: {}", exc)
            return self._empty_fii_dii(str(exc))

    # ------------------------------------------------------------------
    # 2. FII Index Futures Data
    # ------------------------------------------------------------------
    async def get_fii_index_futures(self) -> dict:
        """
        Fetch FII index futures long/short open interest data.

        Returns
        -------
        dict with keys: fii_long, fii_short, fii_net_long_short, long_short_ratio,
                        bias, date, source
        """
        try:
            data = await self._nse_get(self.FII_DERIVATIVES_URL)

            if not data:
                logger.warning("No FII derivatives data returned.")
                return self._empty_fii_futures("No data from NSE")

            # Parse participant-wise OI data
            # NSE returns nested structure; extract FII/FPI Index Futures row
            fii_long = 0.0
            fii_short = 0.0
            trade_date = ""

            if isinstance(data, dict):
                # Might be under a 'data' key or at top level
                oi_data = data.get("data", data)
                trade_date = data.get("date", data.get("timestamp", ""))

                if isinstance(oi_data, list):
                    for row in oi_data:
                        client_type = str(row.get("client_type", row.get("clientType", ""))).upper()
                        if "FII" in client_type or "FPI" in client_type:
                            # Index futures
                            fut_long = self._parse_number(
                                row.get("future_index_long", row.get("futIdxLong", 0))
                            )
                            fut_short = self._parse_number(
                                row.get("future_index_short", row.get("futIdxShort", 0))
                            )
                            fii_long += fut_long
                            fii_short += fut_short
            elif isinstance(data, list):
                for row in data:
                    client_type = str(row.get("client_type", row.get("clientType", ""))).upper()
                    if "FII" in client_type or "FPI" in client_type:
                        fii_long += self._parse_number(
                            row.get("future_index_long", row.get("futIdxLong", 0))
                        )
                        fii_short += self._parse_number(
                            row.get("future_index_short", row.get("futIdxShort", 0))
                        )

            net_ls = fii_long - fii_short
            ratio = round(fii_long / fii_short, 4) if fii_short > 0 else 0.0

            if net_ls > 0:
                bias = "LONG"
            elif net_ls < 0:
                bias = "SHORT"
            else:
                bias = "NEUTRAL"

            result = {
                "fii_long": round(fii_long, 2),
                "fii_short": round(fii_short, 2),
                "fii_net_long_short": round(net_ls, 2),
                "long_short_ratio": ratio,
                "bias": bias,
                "date": trade_date,
                "source": "NSE",
            }

            logger.info(
                "FII futures | long={:.0f} short={:.0f} net={:.0f} ratio={:.2f} bias={}",
                fii_long, fii_short, net_ls, ratio, bias,
            )
            return result

        except Exception as exc:
            logger.error("get_fii_index_futures failed: {}", exc)
            return self._empty_fii_futures(str(exc))

    # ------------------------------------------------------------------
    # 3. Combined latest data
    # ------------------------------------------------------------------
    async def get_latest(self) -> dict:
        """
        Fetch and combine cash market FII/DII data with derivatives FII data.

        Returns
        -------
        dict with keys: cash (fii/dii data), derivatives (futures data), timestamp
        """
        try:
            cash = await self.fetch_fii_dii_data()
            derivatives = await self.get_fii_index_futures()

            result = {
                "cash": cash,
                "derivatives": derivatives,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("FII combined data fetched successfully.")
            return result

        except Exception as exc:
            logger.error("get_latest failed: {}", exc)
            return {
                "cash": self._empty_fii_dii(str(exc)),
                "derivatives": self._empty_fii_futures(str(exc)),
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 4. Scoring
    # ------------------------------------------------------------------
    def score(self, fii_data: dict) -> dict:
        """
        Score institutional flow out of 8 based on:
          - FII net buyer in cash market: +3
          - FII long bias in index futures: +3
          - DII net buyer (supporting): +2

        Parameters
        ----------
        fii_data : dict returned by get_latest() with 'cash' and 'derivatives' keys.

        Returns
        -------
        dict with keys: score, max_score, bias, fii_net, dii_net, details
        """
        try:
            total_score = 0
            details = []

            # Extract sub-dicts
            cash = fii_data.get("cash", {})
            derivatives = fii_data.get("derivatives", {})

            fii_net = cash.get("fii_net", 0)
            dii_net = cash.get("dii_net", 0)

            # --- FII net buyer in cash: +3 ---
            if fii_net > 0:
                total_score += 3
                details.append(f"FII net buyer in cash: +{fii_net:.0f} cr (+3)")
            elif fii_net > -500:
                total_score += 1
                details.append(f"FII marginal seller: {fii_net:.0f} cr (+1)")
            else:
                details.append(f"FII net seller in cash: {fii_net:.0f} cr (+0)")

            # --- FII long bias in futures: +3 ---
            futures_bias = derivatives.get("bias", "NEUTRAL")
            net_ls = derivatives.get("fii_net_long_short", 0)
            ratio = derivatives.get("long_short_ratio", 1.0)

            if futures_bias == "LONG" and ratio >= 1.2:
                total_score += 3
                details.append(f"FII strong long in futures: ratio={ratio:.2f} (+3)")
            elif futures_bias == "LONG":
                total_score += 2
                details.append(f"FII mild long in futures: ratio={ratio:.2f} (+2)")
            elif futures_bias == "NEUTRAL":
                total_score += 1
                details.append(f"FII neutral in futures (+1)")
            else:
                details.append(f"FII short in futures: ratio={ratio:.2f} (+0)")

            # --- DII supporting: +2 ---
            if dii_net > 0:
                total_score += 2
                details.append(f"DII net buyer: +{dii_net:.0f} cr (+2)")
            elif dii_net > -500:
                total_score += 1
                details.append(f"DII marginal seller: {dii_net:.0f} cr (+1)")
            else:
                details.append(f"DII net seller: {dii_net:.0f} cr (+0)")

            # Overall bias
            if total_score >= 6:
                bias = "STRONG BULLISH"
            elif total_score >= 4:
                bias = "BULLISH"
            elif total_score >= 3:
                bias = "NEUTRAL"
            elif total_score >= 2:
                bias = "BEARISH"
            else:
                bias = "STRONG BEARISH"

            result = {
                "score": total_score,
                "max_score": 8,
                "bias": bias,
                "fii_net": round(fii_net, 2),
                "dii_net": round(dii_net, 2),
                "details": details,
            }

            logger.info(
                "FII score: {}/{} | bias={} | FII net={:.0f} | DII net={:.0f}",
                total_score, 8, bias, fii_net, dii_net,
            )
            return result

        except Exception as exc:
            logger.error("FII scoring failed: {}", exc)
            return {
                "score": 0,
                "max_score": 8,
                "bias": "UNKNOWN",
                "fii_net": 0,
                "dii_net": 0,
                "details": [f"Scoring error: {exc}"],
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_number(val) -> float:
        """Safely parse a number that might be a string with commas."""
        if val is None:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        try:
            return float(str(val).replace(",", "").replace(" ", ""))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _empty_fii_dii(error_msg: str = "") -> dict:
        return {
            "fii_buy": 0.0,
            "fii_sell": 0.0,
            "fii_net": 0.0,
            "dii_buy": 0.0,
            "dii_sell": 0.0,
            "dii_net": 0.0,
            "date": "",
            "source": "NSE",
            "raw_data": None,
            "error": error_msg,
        }

    @staticmethod
    def _empty_fii_futures(error_msg: str = "") -> dict:
        return {
            "fii_long": 0.0,
            "fii_short": 0.0,
            "fii_net_long_short": 0.0,
            "long_short_ratio": 0.0,
            "bias": "UNKNOWN",
            "date": "",
            "source": "NSE",
            "error": error_msg,
        }
