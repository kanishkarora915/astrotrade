"""
AstroEngine - Vedic astrology scoring and forecasting engine for market analysis.
Consumes AstroFeed snapshots and produces actionable scores and forecasts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger

import sys
import os

# Ensure parent package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.astro_feed import (
    AstroFeed,
    BENEFIC_PLANETS,
    MALEFIC_PLANETS,
)


# Signs where Jupiter is strong (own sign, exalted, mulatrikona)
JUPITER_BENEFIC_SIGNS = {"Sagittarius", "Pisces", "Cancer"}

BIAS_THRESHOLDS = {
    "strong_bullish": 16,
    "bullish": 12,
    "neutral": 8,
    "bearish": 4,
    "strong_bearish": 0,
}


class AstroEngine:
    """Scores astro snapshots and produces weekly forecasts."""

    def __init__(self):
        self.feed = AstroFeed()
        logger.info("AstroEngine initialized")

    # ------------------------------------------------------------------
    # 1. Score
    # ------------------------------------------------------------------
    def score(self, astro_snapshot: Dict) -> Dict:
        """
        Score an astro snapshot out of 20 points.

        Scoring rubric:
          - Jupiter direct + benefic sign:  +5
          - Nakshatra nature bullish: +5, bearish: -5, volatile: 0
          - Hora benefic:  +4
          - Tithi bullish:  +4
          - No malefic aspects:  +2

        Returns dict with score (0-20), bias, and supporting details.
        """
        try:
            total_score = 0
            key_signals: List[str] = []

            positions = astro_snapshot.get("positions", {})
            nakshatra = astro_snapshot.get("nakshatra", {})
            tithi = astro_snapshot.get("tithi", {})
            hora = astro_snapshot.get("hora", {})
            aspects = astro_snapshot.get("aspects", [])

            # --- Jupiter direct + benefic sign: +5 ---
            jupiter = positions.get("jupiter", {})
            jupiter_retro = jupiter.get("retrograde", True)
            jupiter_sign = jupiter.get("sign", "")
            jupiter_direct = not jupiter_retro

            if jupiter_direct and jupiter_sign in JUPITER_BENEFIC_SIGNS:
                total_score += 5
                key_signals.append(f"Jupiter direct in {jupiter_sign} (+5)")
            elif jupiter_direct:
                total_score += 2
                key_signals.append(f"Jupiter direct but in {jupiter_sign} (+2)")
            else:
                key_signals.append(f"Jupiter retrograde in {jupiter_sign} (+0)")

            # --- Nakshatra nature: bullish +5, bearish -5, volatile 0 ---
            nak_nature = nakshatra.get("nature", "neutral")
            nak_name = nakshatra.get("nakshatra", "Unknown")

            if nak_nature == "bullish":
                total_score += 5
                key_signals.append(f"Moon in {nak_name} nakshatra - bullish (+5)")
            elif nak_nature == "bearish":
                # Don't go below 0 later, but track the deduction
                total_score -= 5
                key_signals.append(f"Moon in {nak_name} nakshatra - bearish (-5)")
            elif nak_nature == "volatile":
                key_signals.append(f"Moon in {nak_name} nakshatra - volatile (+0)")
            else:
                total_score += 2
                key_signals.append(f"Moon in {nak_name} nakshatra - neutral (+2)")

            # --- Hora benefic: +4 ---
            hora_benefic = hora.get("is_benefic", False)
            hora_planet = hora.get("hora_planet", "Unknown")

            if hora_benefic:
                total_score += 4
                key_signals.append(f"Hora of {hora_planet} - benefic (+4)")
            else:
                key_signals.append(f"Hora of {hora_planet} - malefic (+0)")

            # --- Tithi bullish: +4 ---
            tithi_nature = tithi.get("nature", "neutral")
            tithi_name = tithi.get("name", "Unknown")

            if tithi_nature == "bullish":
                total_score += 4
                key_signals.append(f"Tithi {tithi_name} - bullish (+4)")
            elif tithi_nature == "bearish":
                total_score -= 2
                key_signals.append(f"Tithi {tithi_name} - bearish (-2)")
            elif tithi_nature == "volatile":
                total_score += 1
                key_signals.append(f"Tithi {tithi_name} - volatile (+1)")
            else:
                total_score += 2
                key_signals.append(f"Tithi {tithi_name} - neutral (+2)")

            # --- No malefic aspects: +2 ---
            malefic_aspects = [
                a for a in aspects
                if a.get("market_impact") in ("volatile", "reversal", "strongly_bearish")
                and a.get("strength", 0) > 0.5
            ]

            if not malefic_aspects:
                total_score += 2
                key_signals.append("No strong malefic aspects (+2)")
            else:
                malefic_desc = "; ".join(
                    f"{a['planet1']}-{a['planet2']} {a['type']}"
                    for a in malefic_aspects[:3]
                )
                key_signals.append(f"Malefic aspects present: {malefic_desc} (+0)")

            # Clamp score to 0-20
            total_score = max(0, min(20, total_score))

            # Determine bias
            if total_score >= BIAS_THRESHOLDS["strong_bullish"]:
                bias = "strong_bullish"
            elif total_score >= BIAS_THRESHOLDS["bullish"]:
                bias = "bullish"
            elif total_score >= BIAS_THRESHOLDS["neutral"]:
                bias = "neutral"
            elif total_score >= BIAS_THRESHOLDS["bearish"]:
                bias = "bearish"
            else:
                bias = "strong_bearish"

            # Window validity: current hora changes every hour
            now = datetime.now()
            next_hora_change = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

            # Next major change: nakshatra transit (~1 day), tithi (~1 day)
            next_change = now + timedelta(hours=6)  # conservative estimate

            result = {
                "score": total_score,
                "max_score": 20,
                "bias": bias,
                "nakshatra": {
                    "name": nak_name,
                    "nature": nak_nature,
                    "pada": nakshatra.get("pada", 1),
                    "lord": nakshatra.get("lord_planet", "Unknown"),
                },
                "tithi": {
                    "name": tithi_name,
                    "nature": tithi_nature,
                    "paksha": tithi.get("paksha", "Unknown"),
                },
                "hora": {
                    "planet": hora_planet,
                    "nature": hora.get("nature", "neutral"),
                },
                "key_signals": key_signals,
                "window_valid_till": next_hora_change.isoformat(),
                "next_change": next_change.isoformat(),
                "timestamp": astro_snapshot.get("timestamp", now.isoformat()),
            }

            logger.info(
                f"Astro score: {total_score}/20 | bias={bias} "
                f"| signals={len(key_signals)}"
            )
            return result

        except Exception as e:
            logger.error(f"score() failed: {e}")
            return {
                "score": 10, "max_score": 20, "bias": "neutral",
                "key_signals": [f"Scoring error: {e}"],
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # 2. Weekly astro forecast
    # ------------------------------------------------------------------
    def get_weekly_astro_forecast(self, start_date: Optional[datetime] = None) -> List[Dict]:
        """
        Pre-compute astro score and summary for each day of the week.
        Returns a list of 7 daily astro summaries starting from start_date.
        """
        try:
            if start_date is None:
                start_date = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

            forecast: List[Dict] = []

            for day_offset in range(7):
                dt = start_date + timedelta(days=day_offset)
                day_name = dt.strftime("%A")
                date_str = dt.strftime("%Y-%m-%d")

                logger.info(f"Computing forecast for {date_str} ({day_name})")

                try:
                    # Get positions for market open time (10:00 AM IST)
                    market_dt = dt.replace(hour=10, minute=0, second=0)
                    positions = self.feed.get_planet_positions(market_dt)

                    moon_deg = positions.get("moon", {}).get("degree", 0.0)
                    sun_deg = positions.get("sun", {}).get("degree", 0.0)

                    nakshatra = self.feed.get_nakshatra(moon_deg)
                    tithi = self.feed.get_tithi(sun_deg, moon_deg)
                    hora = self.feed.get_hora(market_dt)
                    aspects = self.feed.get_planetary_aspects(positions)

                    snapshot = {
                        "timestamp": market_dt.isoformat(),
                        "positions": positions,
                        "nakshatra": nakshatra,
                        "tithi": tithi,
                        "hora": hora,
                        "aspects": aspects,
                    }

                    day_score = self.score(snapshot)

                    # Count benefic vs malefic aspects
                    benefic_aspects = [
                        a for a in aspects
                        if a.get("market_impact") in ("bullish", "mildly_bullish", "strong_trend")
                    ]
                    malefic_aspects_list = [
                        a for a in aspects
                        if a.get("market_impact") in ("volatile", "reversal", "strongly_bearish")
                    ]

                    # Determine if it's a weekend
                    is_weekend = dt.weekday() >= 5
                    is_trading_day = not is_weekend

                    daily_summary = {
                        "date": date_str,
                        "day": day_name,
                        "is_trading_day": is_trading_day,
                        "score": day_score["score"],
                        "bias": day_score["bias"],
                        "nakshatra": nakshatra.get("nakshatra", "Unknown"),
                        "nakshatra_nature": nakshatra.get("nature", "neutral"),
                        "tithi": tithi.get("name", "Unknown"),
                        "tithi_nature": tithi.get("nature", "neutral"),
                        "hora_at_open": hora.get("hora_planet", "Unknown"),
                        "hora_nature": hora.get("nature", "neutral"),
                        "benefic_aspects_count": len(benefic_aspects),
                        "malefic_aspects_count": len(malefic_aspects_list),
                        "key_signals": day_score.get("key_signals", []),
                        "recommendation": self._daily_recommendation(day_score["score"], day_score["bias"]),
                    }

                    forecast.append(daily_summary)

                except Exception as day_err:
                    logger.warning(f"Forecast failed for {date_str}: {day_err}")
                    forecast.append({
                        "date": date_str,
                        "day": day_name,
                        "is_trading_day": dt.weekday() < 5,
                        "score": 10,
                        "bias": "neutral",
                        "error": str(day_err),
                        "recommendation": "No data available - trade with caution",
                    })

            # Summary stats
            trading_days = [d for d in forecast if d.get("is_trading_day", False)]
            if trading_days:
                avg_score = sum(d["score"] for d in trading_days) / len(trading_days)
                best_day = max(trading_days, key=lambda d: d["score"])
                worst_day = min(trading_days, key=lambda d: d["score"])

                logger.info(
                    f"Weekly forecast ready | avg_score={avg_score:.1f} "
                    f"| best={best_day['day']}({best_day['score']}) "
                    f"| worst={worst_day['day']}({worst_day['score']})"
                )

            return forecast

        except Exception as e:
            logger.error(f"get_weekly_astro_forecast failed: {e}")
            return [{"error": str(e)}]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _daily_recommendation(self, score: int, bias: str) -> str:
        """Generate a human-readable trading recommendation from score and bias."""
        if score >= 16:
            return "Strong bullish astro window - favorable for long positions"
        elif score >= 12:
            return "Moderately bullish - consider buying on dips"
        elif score >= 8:
            return "Neutral astro - trade with technical confirmation only"
        elif score >= 4:
            return "Bearish astro - prefer hedged or short positions"
        else:
            return "Strongly bearish astro - avoid aggressive longs, protect capital"
