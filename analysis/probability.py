"""
WeeklyProbabilityEngine - Multi-factor probability computation for Nifty weekly options.

Combines Black-Scholes, Monte Carlo, astro forecasts, and ML regime predictions
into a weighted per-day probability matrix for the upcoming trading week.
"""

import math
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
from loguru import logger


class WeeklyProbabilityEngine:
    """
    Generates weekly call/put probability forecasts using a weighted blend of:
      - Historical base rates (30%)
      - Astro pre-calculation (25%)
      - Black-Scholes implied probability (25%)
      - ML regime prediction (20%)
    """

    WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Default historical base-rate bias per weekday (neutral starting point)
    HISTORICAL_BASE = {
        "Monday":    {"call": 0.52, "put": 0.48},
        "Tuesday":   {"call": 0.50, "put": 0.50},
        "Wednesday": {"call": 0.49, "put": 0.51},
        "Thursday":  {"call": 0.51, "put": 0.49},
        "Friday":    {"call": 0.48, "put": 0.52},
    }

    # Weights for the four components
    W_HISTORICAL = 0.30
    W_ASTRO = 0.25
    W_BS = 0.25
    W_ML = 0.20

    # ------------------------------------------------------------------
    # Black-Scholes probability
    # ------------------------------------------------------------------
    def black_scholes_probability(
        self,
        spot: float,
        strike: float,
        days: float,
        iv: float,
        rate: float = 0.065,
    ) -> dict:
        """
        Compute ITM probabilities for call and put using the Black-Scholes d2 term.

        Parameters
        ----------
        spot   : Current underlying price.
        strike : Option strike price.
        days   : Calendar days to expiry.
        iv     : Implied volatility (annualised, e.g. 0.15 for 15%).
        rate   : Risk-free rate (default 6.5% for India).

        Returns
        -------
        dict with keys: call_itm_prob, put_itm_prob, d2, spot, strike, days, iv
        """
        try:
            if days <= 0:
                logger.warning("Days to expiry <= 0, returning 50/50 neutral.")
                return {
                    "call_itm_prob": 0.50,
                    "put_itm_prob": 0.50,
                    "d2": 0.0,
                    "spot": spot,
                    "strike": strike,
                    "days": days,
                    "iv": iv,
                }

            T = days / 365.0
            sigma = iv
            sqrt_T = math.sqrt(T)

            d2 = (math.log(spot / strike) + (rate - 0.5 * sigma ** 2) * T) / (
                sigma * sqrt_T
            )

            call_itm = float(norm.cdf(d2))
            put_itm = float(norm.cdf(-d2))

            result = {
                "call_itm_prob": round(call_itm, 6),
                "put_itm_prob": round(put_itm, 6),
                "d2": round(d2, 6),
                "spot": spot,
                "strike": strike,
                "days": days,
                "iv": iv,
            }
            logger.debug(
                "BS prob | spot={} strike={} days={} iv={} => call={:.4f} put={:.4f}",
                spot, strike, days, iv, call_itm, put_itm,
            )
            return result

        except Exception as exc:
            logger.error("Black-Scholes probability failed: {}", exc)
            return {
                "call_itm_prob": 0.50,
                "put_itm_prob": 0.50,
                "d2": 0.0,
                "spot": spot,
                "strike": strike,
                "days": days,
                "iv": iv,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Monte Carlo simulation
    # ------------------------------------------------------------------
    def monte_carlo_probability(
        self,
        spot: float,
        iv: float,
        days: float,
        simulations: int = 10_000,
        rate: float = 0.065,
    ) -> dict:
        """
        Run Geometric Brownian Motion Monte Carlo to estimate the probability
        distribution of outcomes over *days* calendar days.

        Returns
        -------
        dict with keys: call_prob, put_prob, expected_range,
                        mean_price, median_price, std_price,
                        percentile_5, percentile_95, simulations
        """
        try:
            if days <= 0:
                logger.warning("MC: days <= 0, returning neutral.")
                return {
                    "call_prob": 0.50,
                    "put_prob": 0.50,
                    "expected_range": {"low": spot, "high": spot},
                    "mean_price": spot,
                    "median_price": spot,
                    "std_price": 0.0,
                    "percentile_5": spot,
                    "percentile_95": spot,
                    "simulations": simulations,
                }

            T = days / 365.0
            dt = T  # single-step GBM for the total period
            sigma = iv

            np.random.seed(None)  # fresh entropy each call
            Z = np.random.standard_normal(simulations)

            # GBM terminal price: S * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
            drift = (rate - 0.5 * sigma ** 2) * T
            diffusion = sigma * math.sqrt(T) * Z
            terminal_prices = spot * np.exp(drift + diffusion)

            mean_price = float(np.mean(terminal_prices))
            median_price = float(np.median(terminal_prices))
            std_price = float(np.std(terminal_prices))
            p5 = float(np.percentile(terminal_prices, 5))
            p95 = float(np.percentile(terminal_prices, 95))

            call_prob = float(np.mean(terminal_prices > spot))
            put_prob = float(np.mean(terminal_prices < spot))

            result = {
                "call_prob": round(call_prob, 6),
                "put_prob": round(put_prob, 6),
                "expected_range": {
                    "low": round(p5, 2),
                    "high": round(p95, 2),
                },
                "mean_price": round(mean_price, 2),
                "median_price": round(median_price, 2),
                "std_price": round(std_price, 2),
                "percentile_5": round(p5, 2),
                "percentile_95": round(p95, 2),
                "simulations": simulations,
            }
            logger.debug(
                "MC | spot={} iv={} days={} sims={} => call={:.4f} put={:.4f} range=[{:.0f}, {:.0f}]",
                spot, iv, days, simulations, call_prob, put_prob, p5, p95,
            )
            return result

        except Exception as exc:
            logger.error("Monte Carlo simulation failed: {}", exc)
            return {
                "call_prob": 0.50,
                "put_prob": 0.50,
                "expected_range": {"low": spot, "high": spot},
                "mean_price": spot,
                "median_price": spot,
                "std_price": 0.0,
                "percentile_5": spot,
                "percentile_95": spot,
                "simulations": simulations,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Helpers: resolve astro / ML inputs into per-day probabilities
    # ------------------------------------------------------------------
    @staticmethod
    def _astro_day_probability(day_forecast: dict | None) -> dict:
        """
        Convert an astro forecast entry for a single day into call/put probability.
        Expected keys in day_forecast: score (0-10), bias ('bullish'/'bearish'/'neutral').
        """
        if not day_forecast:
            return {"call": 0.50, "put": 0.50}

        score = day_forecast.get("score", 5)
        bias = day_forecast.get("bias", "neutral").lower()

        # Map score 0-10 to probability tilt
        # score 10 => strong bullish (call 0.80), score 0 => strong bearish (call 0.20)
        call_p = 0.20 + (score / 10.0) * 0.60  # range [0.20, 0.80]

        if bias == "bearish":
            call_p = min(call_p, 0.45)
        elif bias == "bullish":
            call_p = max(call_p, 0.55)

        return {"call": round(call_p, 4), "put": round(1.0 - call_p, 4)}

    @staticmethod
    def _ml_day_probability(day_prediction: dict | None) -> dict:
        """
        Convert an ML prediction for a single day into call/put probability.
        Expected keys: regime ('bull'/'bear'/'sideways'), confidence (0-1).
        """
        if not day_prediction:
            return {"call": 0.50, "put": 0.50}

        regime = day_prediction.get("regime", "sideways").lower()
        confidence = day_prediction.get("confidence", 0.50)
        confidence = max(0.0, min(1.0, confidence))

        if regime == "bull":
            call_p = 0.50 + 0.30 * confidence  # up to 0.80
        elif regime == "bear":
            call_p = 0.50 - 0.30 * confidence  # down to 0.20
        else:
            call_p = 0.50

        return {"call": round(call_p, 4), "put": round(1.0 - call_p, 4)}

    # ------------------------------------------------------------------
    # Main weekly computation
    # ------------------------------------------------------------------
    def compute_next_week(
        self,
        spot: float,
        iv: float,
        astro_forecast: list[dict] | None = None,
        ml_prediction: list[dict] | None = None,
    ) -> dict:
        """
        Compute per-day probability matrix for the next trading week (Mon-Fri).

        Parameters
        ----------
        spot           : Current Nifty spot price.
        iv             : Annualised implied volatility (e.g. 0.13 for 13%).
        astro_forecast : Optional list of 5 dicts (one per trading day) with keys:
                         date, day, nakshatra, tithi, score (0-10), bias, risk_events.
        ml_prediction  : Optional list of 5 dicts with keys:
                         date, regime ('bull'/'bear'/'sideways'), confidence (0-1).

        Returns
        -------
        dict with keys: week_start, week_end, spot, iv, days (list of 5 day dicts).
        Each day dict: date, day, nakshatra, tithi, call_probability, put_probability,
                       bias, confidence, best_trade, risk_events, astro_score,
                       components (breakdown of each factor).
        """
        try:
            # Determine next Monday
            today = datetime.now()
            days_until_monday = (7 - today.weekday()) % 7
            if days_until_monday == 0 and today.weekday() != 0:
                days_until_monday = 7
            if today.weekday() == 0:
                days_until_monday = 0
            next_monday = today + timedelta(days=days_until_monday)

            # Normalise / default astro & ML inputs
            if astro_forecast is None:
                astro_forecast = [None] * 5
            if ml_prediction is None:
                ml_prediction = [None] * 5

            # Pad to 5 if shorter
            astro_forecast = list(astro_forecast) + [None] * (5 - len(astro_forecast))
            ml_prediction = list(ml_prediction) + [None] * (5 - len(ml_prediction))

            week_days = []

            for i, day_name in enumerate(self.WEEKDAYS):
                day_date = next_monday + timedelta(days=i)
                days_to_expiry = 5 - i  # rough calendar days until weekly expiry (Thu)

                # --- Component 1: Historical base rate (30%) ---
                hist = self.HISTORICAL_BASE[day_name]
                hist_call = hist["call"]
                hist_put = hist["put"]

                # --- Component 2: Astro pre-calculation (25%) ---
                astro_entry = astro_forecast[i] if i < len(astro_forecast) else None
                astro_prob = self._astro_day_probability(astro_entry)
                astro_call = astro_prob["call"]
                astro_put = astro_prob["put"]

                # --- Component 3: Black-Scholes probability (25%) ---
                atm_strike = round(spot / 50) * 50  # nearest 50 strike
                bs = self.black_scholes_probability(
                    spot=spot,
                    strike=atm_strike,
                    days=max(days_to_expiry, 0.5),
                    iv=iv,
                )
                bs_call = bs["call_itm_prob"]
                bs_put = bs["put_itm_prob"]

                # --- Component 4: ML regime prediction (20%) ---
                ml_entry = ml_prediction[i] if i < len(ml_prediction) else None
                ml_prob = self._ml_day_probability(ml_entry)
                ml_call = ml_prob["call"]
                ml_put = ml_prob["put"]

                # --- Weighted combination ---
                combined_call = (
                    self.W_HISTORICAL * hist_call
                    + self.W_ASTRO * astro_call
                    + self.W_BS * bs_call
                    + self.W_ML * ml_call
                )
                combined_put = (
                    self.W_HISTORICAL * hist_put
                    + self.W_ASTRO * astro_put
                    + self.W_BS * bs_put
                    + self.W_ML * ml_put
                )

                # Normalise so they sum to 1
                total = combined_call + combined_put
                if total > 0:
                    combined_call /= total
                    combined_put /= total

                # Determine bias & confidence
                diff = abs(combined_call - combined_put)
                if combined_call > combined_put + 0.05:
                    bias = "BULLISH"
                elif combined_put > combined_call + 0.05:
                    bias = "BEARISH"
                else:
                    bias = "NEUTRAL"

                # Confidence: how far from 50/50 (scaled 0-100)
                confidence = round(min(diff * 2, 1.0) * 100, 1)

                # Best trade suggestion
                if bias == "BULLISH" and confidence > 30:
                    best_trade = "Buy CE / Bull Call Spread"
                elif bias == "BEARISH" and confidence > 30:
                    best_trade = "Buy PE / Bear Put Spread"
                elif bias == "NEUTRAL":
                    best_trade = "Iron Condor / Short Straddle"
                else:
                    best_trade = "Wait / Small position"

                # Extract astro metadata
                nakshatra = (astro_entry or {}).get("nakshatra", "N/A")
                tithi = (astro_entry or {}).get("tithi", "N/A")
                risk_events = (astro_entry or {}).get("risk_events", [])
                astro_score = (astro_entry or {}).get("score", 5)

                day_result = {
                    "date": day_date.strftime("%Y-%m-%d"),
                    "day": day_name,
                    "nakshatra": nakshatra,
                    "tithi": tithi,
                    "call_probability": round(combined_call, 4),
                    "put_probability": round(combined_put, 4),
                    "bias": bias,
                    "confidence": confidence,
                    "best_trade": best_trade,
                    "risk_events": risk_events,
                    "astro_score": astro_score,
                    "components": {
                        "historical": {"call": round(hist_call, 4), "put": round(hist_put, 4), "weight": self.W_HISTORICAL},
                        "astro": {"call": round(astro_call, 4), "put": round(astro_put, 4), "weight": self.W_ASTRO},
                        "black_scholes": {"call": round(bs_call, 4), "put": round(bs_put, 4), "weight": self.W_BS},
                        "ml_regime": {"call": round(ml_call, 4), "put": round(ml_put, 4), "weight": self.W_ML},
                    },
                }
                week_days.append(day_result)

            week_end = next_monday + timedelta(days=4)

            result = {
                "week_start": next_monday.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
                "spot": spot,
                "iv": iv,
                "days": week_days,
            }

            logger.info(
                "Weekly probability computed | {} to {} | spot={} iv={}",
                result["week_start"], result["week_end"], spot, iv,
            )
            return result

        except Exception as exc:
            logger.error("compute_next_week failed: {}", exc)
            return {
                "week_start": None,
                "week_end": None,
                "spot": spot,
                "iv": iv,
                "days": [],
                "error": str(exc),
            }
