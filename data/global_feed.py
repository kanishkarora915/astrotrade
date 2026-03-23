"""
GlobalFeed - Fetches global market cues: Gift Nifty, US futures, commodities.

Scores the global environment for Indian market direction.
"""

import httpx
from datetime import datetime
from loguru import logger

# Config import for API keys - graceful fallback if not available
try:
    from config import ALPHA_VANTAGE_KEY
except ImportError:
    try:
        from config import ALPHA_VANTAGE_KEY
    except ImportError:
        ALPHA_VANTAGE_KEY = ""
        logger.warning("ALPHA_VANTAGE_KEY not found in config; some feeds may be limited.")


class GlobalFeed:
    """
    Fetches and scores global market data relevant to Nifty trading:
      - Gift Nifty (SGX Nifty replacement) for pre-market direction
      - US market futures (Dow, S&P 500, Nasdaq)
      - Commodities (Gold, Crude, USD/INR)
    """

    ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
    INVESTING_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    TIMEOUT = 15.0

    # Alpha Vantage symbols
    AV_SYMBOLS = {
        "sp500": "SPY",
        "nasdaq": "QQQ",
        "dow": "DIA",
        "gold": "GLD",
        "crude": "USO",
        "usdinr": "USD/INR",
    }

    def __init__(self, alpha_vantage_key: str | None = None):
        self.api_key = alpha_vantage_key or ALPHA_VANTAGE_KEY

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------
    async def _http_get(self, url: str, params: dict | None = None) -> dict | None:
        """Perform an async GET and return JSON or None on failure."""
        try:
            async with httpx.AsyncClient(
                headers=self.INVESTING_HEADERS,
                follow_redirects=True,
                timeout=self.TIMEOUT,
            ) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("HTTP GET {} failed: {}", url, exc)
            return None

    def _sync_http_get(self, url: str, params: dict | None = None) -> dict | None:
        """Synchronous GET wrapper."""
        try:
            with httpx.Client(
                headers=self.INVESTING_HEADERS,
                follow_redirects=True,
                timeout=self.TIMEOUT,
            ) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("Sync HTTP GET {} failed: {}", url, exc)
            return None

    async def _av_quote(self, symbol: str) -> dict | None:
        """Fetch a GLOBAL_QUOTE from Alpha Vantage for the given symbol."""
        if not self.api_key:
            logger.warning("No Alpha Vantage key; skipping quote for {}", symbol)
            return None

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key,
        }
        data = await self._http_get(self.ALPHA_VANTAGE_BASE, params=params)
        if data and "Global Quote" in data:
            return data["Global Quote"]
        return None

    async def _av_fx_rate(self, from_currency: str, to_currency: str) -> dict | None:
        """Fetch currency exchange rate from Alpha Vantage."""
        if not self.api_key:
            return None

        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_currency,
            "to_currency": to_currency,
            "apikey": self.api_key,
        }
        data = await self._http_get(self.ALPHA_VANTAGE_BASE, params=params)
        if data and "Realtime Currency Exchange Rate" in data:
            return data["Realtime Currency Exchange Rate"]
        return None

    @staticmethod
    def _parse_float(val, default: float = 0.0) -> float:
        """Safely parse a float from a possibly-string value."""
        if val is None:
            return default
        try:
            return float(str(val).replace(",", "").replace("%", ""))
        except (ValueError, TypeError):
            return default

    # ------------------------------------------------------------------
    # 1. Gift Nifty
    # ------------------------------------------------------------------
    async def get_gift_nifty(self) -> dict:
        """
        Fetch Gift Nifty (NSE International Exchange, formerly SGX Nifty) data.

        Uses Nifty Futures as a proxy via Alpha Vantage or web data.
        Gift Nifty trades on NSE IX and is the primary pre-market indicator.

        Returns
        -------
        dict with keys: price, change, change_pct, nifty_close_ref, gap_points,
                        gap_pct, direction, timestamp, source
        """
        try:
            # Try fetching Nifty 50 ETF as a proxy via Alpha Vantage
            quote = await self._av_quote("INDA")  # iShares MSCI India ETF as proxy

            if quote:
                price = self._parse_float(quote.get("05. price"))
                prev_close = self._parse_float(quote.get("08. previous close"))
                change = self._parse_float(quote.get("09. change"))
                change_pct = self._parse_float(quote.get("10. change percent"))

                direction = "POSITIVE" if change > 0 else ("NEGATIVE" if change < 0 else "FLAT")

                result = {
                    "price": round(price, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "nifty_close_ref": round(prev_close, 2),
                    "gap_points": round(change, 2),
                    "gap_pct": round(change_pct, 2),
                    "direction": direction,
                    "timestamp": datetime.now().isoformat(),
                    "source": "AlphaVantage_INDA_proxy",
                }
                logger.info("Gift Nifty proxy | price={} change={} dir={}", price, change, direction)
                return result

            # Fallback: return unknown
            logger.warning("Gift Nifty data unavailable.")
            return {
                "price": 0.0,
                "change": 0.0,
                "change_pct": 0.0,
                "nifty_close_ref": 0.0,
                "gap_points": 0.0,
                "gap_pct": 0.0,
                "direction": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "source": "unavailable",
            }

        except Exception as exc:
            logger.error("get_gift_nifty failed: {}", exc)
            return {
                "price": 0.0,
                "change": 0.0,
                "change_pct": 0.0,
                "nifty_close_ref": 0.0,
                "gap_points": 0.0,
                "gap_pct": 0.0,
                "direction": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "source": "error",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 2. US Markets
    # ------------------------------------------------------------------
    async def get_us_markets(self) -> dict:
        """
        Fetch US market futures / ETFs for Dow Jones, S&P 500, Nasdaq.

        Returns
        -------
        dict with keys: dow, sp500, nasdaq, overall_sentiment, timestamp
        Each sub-dict has: price, change, change_pct, direction
        """
        try:
            symbols = {
                "dow": "DIA",
                "sp500": "SPY",
                "nasdaq": "QQQ",
            }

            results = {}
            sentiment_sum = 0

            for name, symbol in symbols.items():
                quote = await self._av_quote(symbol)

                if quote:
                    price = self._parse_float(quote.get("05. price"))
                    change = self._parse_float(quote.get("09. change"))
                    change_pct = self._parse_float(quote.get("10. change percent"))
                    direction = "GREEN" if change > 0 else ("RED" if change < 0 else "FLAT")

                    results[name] = {
                        "price": round(price, 2),
                        "change": round(change, 2),
                        "change_pct": round(change_pct, 2),
                        "direction": direction,
                        "symbol": symbol,
                    }

                    if change > 0:
                        sentiment_sum += 1
                    elif change < 0:
                        sentiment_sum -= 1
                else:
                    results[name] = {
                        "price": 0.0,
                        "change": 0.0,
                        "change_pct": 0.0,
                        "direction": "UNKNOWN",
                        "symbol": symbol,
                    }

            if sentiment_sum >= 2:
                overall = "BULLISH"
            elif sentiment_sum <= -2:
                overall = "BEARISH"
            elif sentiment_sum > 0:
                overall = "MILDLY BULLISH"
            elif sentiment_sum < 0:
                overall = "MILDLY BEARISH"
            else:
                overall = "MIXED"

            results["overall_sentiment"] = overall
            results["timestamp"] = datetime.now().isoformat()

            logger.info("US markets | Dow={} S&P={} Nasdaq={} | sentiment={}",
                        results.get("dow", {}).get("direction"),
                        results.get("sp500", {}).get("direction"),
                        results.get("nasdaq", {}).get("direction"),
                        overall)
            return results

        except Exception as exc:
            logger.error("get_us_markets failed: {}", exc)
            return {
                "dow": {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"},
                "sp500": {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"},
                "nasdaq": {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"},
                "overall_sentiment": "UNKNOWN",
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 3. Commodities
    # ------------------------------------------------------------------
    async def get_commodities(self) -> dict:
        """
        Fetch Gold, Crude Oil, and USD/INR data.

        Returns
        -------
        dict with keys: gold, crude, usdinr, timestamp
        """
        try:
            result = {"timestamp": datetime.now().isoformat()}

            # Gold (GLD ETF proxy)
            gold_quote = await self._av_quote("GLD")
            if gold_quote:
                gold_price = self._parse_float(gold_quote.get("05. price"))
                gold_change = self._parse_float(gold_quote.get("09. change"))
                gold_pct = self._parse_float(gold_quote.get("10. change percent"))
                result["gold"] = {
                    "price": round(gold_price, 2),
                    "change": round(gold_change, 2),
                    "change_pct": round(gold_pct, 2),
                    "direction": "UP" if gold_change > 0 else ("DOWN" if gold_change < 0 else "FLAT"),
                    "symbol": "GLD",
                }
            else:
                result["gold"] = {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"}

            # Crude Oil (USO ETF proxy)
            crude_quote = await self._av_quote("USO")
            if crude_quote:
                crude_price = self._parse_float(crude_quote.get("05. price"))
                crude_change = self._parse_float(crude_quote.get("09. change"))
                crude_pct = self._parse_float(crude_quote.get("10. change percent"))
                result["crude"] = {
                    "price": round(crude_price, 2),
                    "change": round(crude_change, 2),
                    "change_pct": round(crude_pct, 2),
                    "direction": "UP" if crude_change > 0 else ("DOWN" if crude_change < 0 else "FLAT"),
                    "symbol": "USO",
                }
            else:
                result["crude"] = {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"}

            # USD/INR
            fx_data = await self._av_fx_rate("USD", "INR")
            if fx_data:
                fx_rate = self._parse_float(fx_data.get("5. Exchange Rate"))
                result["usdinr"] = {
                    "rate": round(fx_rate, 4),
                    "direction": "UNKNOWN",  # No change data from single FX call
                    "source": "AlphaVantage",
                }
            else:
                result["usdinr"] = {"rate": 0, "direction": "UNKNOWN"}

            logger.info(
                "Commodities | Gold={} Crude={} USD/INR={}",
                result["gold"].get("direction"),
                result["crude"].get("direction"),
                result.get("usdinr", {}).get("rate"),
            )
            return result

        except Exception as exc:
            logger.error("get_commodities failed: {}", exc)
            return {
                "gold": {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"},
                "crude": {"price": 0, "change": 0, "change_pct": 0, "direction": "UNKNOWN"},
                "usdinr": {"rate": 0, "direction": "UNKNOWN"},
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 4. Combined snapshot
    # ------------------------------------------------------------------
    async def get_snapshot(self) -> dict:
        """
        Combine all global cues into a single snapshot.

        Returns
        -------
        dict with keys: gift_nifty, us_markets, commodities, timestamp
        """
        try:
            gift_nifty = await self.get_gift_nifty()
            us_markets = await self.get_us_markets()
            commodities = await self.get_commodities()

            snapshot = {
                "gift_nifty": gift_nifty,
                "us_markets": us_markets,
                "commodities": commodities,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("Global snapshot fetched successfully.")
            return snapshot

        except Exception as exc:
            logger.error("get_snapshot failed: {}", exc)
            return {
                "gift_nifty": {},
                "us_markets": {},
                "commodities": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 5. Scoring
    # ------------------------------------------------------------------
    def score(self, global_data: dict | None = None) -> dict:
        """
        Score global cues out of 7 for Indian market direction:
          - Gift Nifty positive: +2
          - US futures green: +2
          - Gold stable/up (safe haven not panic): +1
          - USD/INR stable: +1
          - Crude not spiking: +1

        Parameters
        ----------
        global_data : dict from get_snapshot(). If None, returns zero score.

        Returns
        -------
        dict with keys: score, max_score, bias, gift_nifty_change, us_sentiment, details
        """
        try:
            if not global_data:
                return {
                    "score": 0,
                    "max_score": 7,
                    "bias": "UNKNOWN",
                    "gift_nifty_change": 0.0,
                    "us_sentiment": "UNKNOWN",
                    "details": ["No global data available"],
                }

            total_score = 0
            details = []

            # --- Gift Nifty positive: +2 ---
            gift = global_data.get("gift_nifty", {})
            gift_change = gift.get("change", 0)
            gift_pct = gift.get("change_pct", 0)
            gift_dir = gift.get("direction", "UNKNOWN")

            if gift_dir == "POSITIVE" or gift_change > 0:
                if gift_pct > 0.5:
                    total_score += 2
                    details.append(f"Gift Nifty strong positive: +{gift_pct:.2f}% (+2)")
                else:
                    total_score += 1
                    details.append(f"Gift Nifty mildly positive: +{gift_pct:.2f}% (+1)")
            elif gift_dir == "FLAT" or abs(gift_change) < 10:
                total_score += 1
                details.append(f"Gift Nifty flat: {gift_pct:.2f}% (+1)")
            else:
                details.append(f"Gift Nifty negative: {gift_pct:.2f}% (+0)")

            # --- US futures green: +2 ---
            us = global_data.get("us_markets", {})
            us_sentiment = us.get("overall_sentiment", "UNKNOWN")

            if us_sentiment in ("BULLISH",):
                total_score += 2
                details.append(f"US markets bullish (+2)")
            elif us_sentiment in ("MILDLY BULLISH",):
                total_score += 1
                details.append(f"US markets mildly bullish (+1)")
            elif us_sentiment == "MIXED":
                total_score += 1
                details.append(f"US markets mixed (+1)")
            else:
                details.append(f"US markets bearish/unknown (+0)")

            # --- Gold stable/up (not panic selling): +1 ---
            commodities = global_data.get("commodities", {})
            gold = commodities.get("gold", {})
            gold_dir = gold.get("direction", "UNKNOWN")
            gold_pct = gold.get("change_pct", 0)

            if gold_dir in ("UP", "FLAT") or abs(gold_pct) < 1.5:
                total_score += 1
                details.append(f"Gold stable/up: {gold_pct:.2f}% (+1)")
            else:
                details.append(f"Gold dropping sharply: {gold_pct:.2f}% (+0)")

            # --- USD/INR stable: +1 ---
            usdinr = commodities.get("usdinr", {})
            usdinr_dir = usdinr.get("direction", "UNKNOWN")
            usdinr_rate = usdinr.get("rate", 0)

            # If we have a rate, assume stable if direction is not strongly depreciating
            # A rising USD/INR (rupee weakening) is negative for markets
            if usdinr_dir in ("UNKNOWN", "FLAT") or usdinr_rate == 0:
                # Assume stable if no strong signal
                total_score += 1
                details.append(f"USD/INR stable: rate={usdinr_rate} (+1)")
            elif usdinr_dir == "DOWN":
                # Rupee strengthening = positive
                total_score += 1
                details.append(f"USD/INR: rupee strengthening (+1)")
            else:
                details.append(f"USD/INR: rupee weakening (+0)")

            # --- Crude not spiking: +1 ---
            crude = commodities.get("crude", {})
            crude_dir = crude.get("direction", "UNKNOWN")
            crude_pct = crude.get("change_pct", 0)

            if crude_pct < 2.0:  # Not spiking (less than 2% up)
                total_score += 1
                details.append(f"Crude stable: {crude_pct:.2f}% (+1)")
            else:
                details.append(f"Crude spiking: +{crude_pct:.2f}% (+0)")

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
                "max_score": 7,
                "bias": bias,
                "gift_nifty_change": round(gift_change, 2),
                "us_sentiment": us_sentiment,
                "details": details,
            }

            logger.info(
                "Global score: {}/{} | bias={} | Gift Nifty={} | US={}",
                total_score, 7, bias, gift_dir, us_sentiment,
            )
            return result

        except Exception as exc:
            logger.error("Global scoring failed: {}", exc)
            return {
                "score": 0,
                "max_score": 7,
                "bias": "UNKNOWN",
                "gift_nifty_change": 0.0,
                "us_sentiment": "UNKNOWN",
                "details": [f"Scoring error: {exc}"],
                "error": str(exc),
            }
