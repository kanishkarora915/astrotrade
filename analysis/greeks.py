"""
AstroNifty Greeks Analyzer
Black-Scholes based IV calculation, Greeks computation, IV Rank/Percentile scoring.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from loguru import logger


class GreeksAnalyzer:
    """Option Greeks and Implied Volatility analysis engine."""

    def __init__(self):
        self.name = "GreeksAnalyzer"
        self.max_iv = 5.0        # 500% cap
        self.min_iv = 0.001      # 0.1% floor
        self.nr_tolerance = 1e-6
        self.nr_max_iter = 100

    # ------------------------------------------------------------------ #
    #  Black-Scholes helpers
    # ------------------------------------------------------------------ #
    def _bs_price(self, spot: float, strike: float, T: float,
                  r: float, sigma: float, option_type: str) -> float:
        """Return Black-Scholes theoretical price."""
        if T <= 0 or sigma <= 0:
            # Intrinsic only
            if option_type.upper() == "CE":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)

        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type.upper() == "CE":
            price = spot * norm.cdf(d1) - strike * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = strike * math.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return price

    def _bs_vega(self, spot: float, strike: float, T: float,
                 r: float, sigma: float) -> float:
        """Vega: dPrice/dSigma (for Newton-Raphson)."""
        if T <= 0 or sigma <= 0:
            return 0.0
        d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return spot * math.sqrt(T) * norm.pdf(d1)

    # ------------------------------------------------------------------ #
    #  1. Implied Volatility via Newton-Raphson
    # ------------------------------------------------------------------ #
    def calculate_iv(self, option_price: float, spot: float, strike: float,
                     days_to_expiry: float, rate: float = 0.065,
                     option_type: str = "CE") -> float:
        """
        Solve for implied volatility using Newton-Raphson on Black-Scholes.

        Parameters
        ----------
        option_price : market price of the option
        spot         : underlying spot price
        strike       : strike price
        days_to_expiry : calendar days to expiry
        rate         : risk-free rate (annualised, default 6.5%)
        option_type  : 'CE' or 'PE'

        Returns
        -------
        Implied volatility as a decimal (e.g. 0.18 = 18%).
        Returns 0.0 on failure.
        """
        try:
            T = days_to_expiry / 365.0
            if T <= 0:
                logger.debug("IV calc: days_to_expiry <= 0, returning 0")
                return 0.0

            # Intrinsic value check
            if option_type.upper() == "CE":
                intrinsic = max(spot - strike, 0.0)
            else:
                intrinsic = max(strike - spot, 0.0)

            if option_price <= intrinsic:
                logger.debug("Option price <= intrinsic, IV ~ 0")
                return 0.0

            # Deep OTM with near-zero premium
            if option_price < 0.05:
                return 0.0

            # Initial guess using Brenner-Subrahmanyam approximation
            sigma = math.sqrt(2 * math.pi / T) * (option_price / spot)
            sigma = max(self.min_iv, min(sigma, self.max_iv))

            for i in range(self.nr_max_iter):
                bs_price = self._bs_price(spot, strike, T, rate, sigma, option_type)
                vega = self._bs_vega(spot, strike, T, rate, sigma)

                if vega < 1e-12:
                    # Vega too small, try bisection fallback
                    return self._iv_bisection(option_price, spot, strike, T, rate, option_type)

                diff = bs_price - option_price
                sigma -= diff / vega

                sigma = max(self.min_iv, min(sigma, self.max_iv))

                if abs(diff) < self.nr_tolerance:
                    return round(sigma, 6)

            logger.warning(f"IV Newton-Raphson did not converge for "
                           f"S={spot} K={strike} P={option_price}, last sigma={sigma:.4f}")
            return round(sigma, 6)

        except Exception as e:
            logger.error(f"calculate_iv error: {e}")
            return 0.0

    def _iv_bisection(self, option_price: float, spot: float, strike: float,
                      T: float, rate: float, option_type: str) -> float:
        """Bisection fallback when Newton-Raphson vega is too small."""
        lo, hi = self.min_iv, self.max_iv
        for _ in range(200):
            mid = (lo + hi) / 2.0
            price = self._bs_price(spot, strike, T, rate, mid, option_type)
            if abs(price - option_price) < self.nr_tolerance:
                return round(mid, 6)
            if price < option_price:
                lo = mid
            else:
                hi = mid
        return round((lo + hi) / 2.0, 6)

    # ------------------------------------------------------------------ #
    #  2. Greeks calculation
    # ------------------------------------------------------------------ #
    def calculate_greeks(self, spot: float, strike: float, days: float,
                         iv: float, rate: float = 0.065,
                         option_type: str = "CE") -> dict:
        """
        Compute Delta, Gamma, Theta, Vega from Black-Scholes.

        Returns dict with keys: delta, gamma, theta, vega
        """
        try:
            T = days / 365.0
            result = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

            if T <= 0 or iv <= 0:
                # At expiry delta is binary
                if option_type.upper() == "CE":
                    result["delta"] = 1.0 if spot > strike else 0.0
                else:
                    result["delta"] = -1.0 if spot < strike else 0.0
                return result

            sqrt_T = math.sqrt(T)
            d1 = (math.log(spot / strike) + (rate + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
            d2 = d1 - iv * sqrt_T

            nd1 = norm.cdf(d1)
            nd2 = norm.cdf(d2)
            npd1 = norm.pdf(d1)  # standard normal PDF at d1

            # --- Delta ---
            if option_type.upper() == "CE":
                delta = nd1
            else:
                delta = nd1 - 1.0

            # --- Gamma (same for CE and PE) ---
            gamma = npd1 / (spot * iv * sqrt_T)

            # --- Theta (per calendar day) ---
            common_theta = -(spot * npd1 * iv) / (2 * sqrt_T)
            if option_type.upper() == "CE":
                theta = common_theta - rate * strike * math.exp(-rate * T) * nd2
            else:
                theta = common_theta + rate * strike * math.exp(-rate * T) * norm.cdf(-d2)
            theta_per_day = theta / 365.0

            # --- Vega (per 1% move in IV) ---
            vega = spot * sqrt_T * npd1 / 100.0

            result = {
                "delta": round(delta, 6),
                "gamma": round(gamma, 6),
                "theta": round(theta_per_day, 4),
                "vega": round(vega, 4),
            }
            return result

        except Exception as e:
            logger.error(f"calculate_greeks error: {e}")
            return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    # ------------------------------------------------------------------ #
    #  3. IV Rank (52-week)
    # ------------------------------------------------------------------ #
    def calculate_iv_rank(self, current_iv: float, iv_history: list) -> float:
        """
        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100

        Parameters
        ----------
        current_iv : current implied volatility
        iv_history : list of historical IV values (ideally 252 trading days)

        Returns
        -------
        IV Rank as a percentage 0-100.
        """
        try:
            if not iv_history or len(iv_history) < 2:
                logger.warning("iv_history too short for IV Rank calculation")
                return 50.0  # neutral fallback

            clean = [v for v in iv_history if v is not None and v > 0]
            if len(clean) < 2:
                return 50.0

            iv_low = min(clean)
            iv_high = max(clean)

            if iv_high == iv_low:
                return 50.0

            rank = (current_iv - iv_low) / (iv_high - iv_low) * 100.0
            return round(max(0.0, min(100.0, rank)), 2)

        except Exception as e:
            logger.error(f"calculate_iv_rank error: {e}")
            return 50.0

    # ------------------------------------------------------------------ #
    #  4. IV Percentile
    # ------------------------------------------------------------------ #
    def calculate_iv_percentile(self, current_iv: float, iv_history: list) -> float:
        """
        IV Percentile = % of days where IV was BELOW current level.

        Parameters
        ----------
        current_iv : current implied volatility
        iv_history : list of historical IV values

        Returns
        -------
        IV Percentile as 0-100.
        """
        try:
            if not iv_history or len(iv_history) < 2:
                logger.warning("iv_history too short for IV Percentile")
                return 50.0

            clean = [v for v in iv_history if v is not None and v > 0]
            if len(clean) < 2:
                return 50.0

            below_count = sum(1 for v in clean if v < current_iv)
            percentile = (below_count / len(clean)) * 100.0
            return round(max(0.0, min(100.0, percentile)), 2)

        except Exception as e:
            logger.error(f"calculate_iv_percentile error: {e}")
            return 50.0

    # ------------------------------------------------------------------ #
    #  5. Composite Greeks Score
    # ------------------------------------------------------------------ #
    def score(self, chain: pd.DataFrame, spot: float) -> dict:
        """
        Score options chain from a Greeks perspective (out of 10).

        Scoring:
          - IV Rank < 30 (cheap options, good for buying) : +3
          - ATM Delta favorable for direction              : +3
          - Gamma not extreme (not expiry trap)            : +2
          - Theta decay manageable                         : +2

        Parameters
        ----------
        chain : DataFrame with columns like strike, iv, oi, ltp, option_type, etc.
        spot  : current spot price

        Returns
        -------
        dict with: score, bias, iv_rank, atm_delta, details
        """
        try:
            result = {
                "score": 5.0,
                "bias": "NEUTRAL",
                "iv_rank": 50.0,
                "atm_delta": 0.5,
                "details": [],
            }

            if chain is None or chain.empty:
                result["details"].append("No chain data available")
                return result

            score = 0.0
            details = []

            # ---- Find ATM strike ----
            strikes = chain["strike"].unique() if "strike" in chain.columns else []
            if len(strikes) == 0:
                result["details"].append("No strikes in chain")
                return result

            atm_strike = min(strikes, key=lambda s: abs(s - spot))

            # ---- Compute IV from chain (use 'iv' column or calculate) ----
            iv_col = None
            for candidate in ["iv", "implied_volatility", "impliedVolatility"]:
                if candidate in chain.columns:
                    iv_col = candidate
                    break

            iv_values = []
            current_iv = 0.0
            if iv_col:
                iv_values = chain[iv_col].dropna().tolist()
                atm_rows = chain[chain["strike"] == atm_strike]
                if not atm_rows.empty:
                    current_iv = atm_rows[iv_col].mean()

            # ---- 1. IV Rank scoring (+3) ----
            iv_rank = self.calculate_iv_rank(current_iv, iv_values) if iv_values else 50.0
            result["iv_rank"] = iv_rank

            if iv_rank < 30:
                score += 3.0
                details.append(f"IV Rank {iv_rank:.1f} < 30 => cheap options, favor buying (+3)")
            elif iv_rank < 50:
                score += 2.0
                details.append(f"IV Rank {iv_rank:.1f} moderate (+2)")
            elif iv_rank < 70:
                score += 1.0
                details.append(f"IV Rank {iv_rank:.1f} elevated (+1)")
            else:
                score += 0.0
                details.append(f"IV Rank {iv_rank:.1f} >= 70 => expensive options (+0)")

            # ---- 2. ATM Delta for directional bias (+3) ----
            atm_ce = chain[(chain["strike"] == atm_strike)]
            if "option_type" in chain.columns:
                atm_ce = chain[(chain["strike"] == atm_strike) &
                               (chain["option_type"].str.upper() == "CE")]

            atm_delta = 0.5
            if not atm_ce.empty and current_iv > 0:
                days_col = None
                for c in ["days_to_expiry", "dte", "days"]:
                    if c in chain.columns:
                        days_col = c
                        break
                dte = atm_ce[days_col].iloc[0] if days_col else 7
                greeks = self.calculate_greeks(spot, atm_strike, dte, current_iv)
                atm_delta = greeks["delta"]

            result["atm_delta"] = round(atm_delta, 4)

            if atm_delta > 0.55:
                score += 3.0
                bias = "BULLISH"
                details.append(f"ATM CE Delta {atm_delta:.3f} > 0.55 => bullish lean (+3)")
            elif atm_delta < 0.45:
                score += 1.0
                bias = "BEARISH"
                details.append(f"ATM CE Delta {atm_delta:.3f} < 0.45 => bearish lean (+1)")
            else:
                score += 2.0
                bias = "NEUTRAL"
                details.append(f"ATM CE Delta {atm_delta:.3f} near 0.50 => neutral (+2)")

            # ---- 3. Gamma check (not extreme / expiry trap) (+2) ----
            if not atm_ce.empty and current_iv > 0:
                days_col_val = None
                for c in ["days_to_expiry", "dte", "days"]:
                    if c in chain.columns:
                        days_col_val = c
                        break
                dte_val = atm_ce[days_col_val].iloc[0] if days_col_val else 7
                greeks = self.calculate_greeks(spot, atm_strike, dte_val, current_iv)
                gamma = greeks["gamma"]

                # Gamma relative to spot; extreme if gamma*spot > 0.05
                gamma_impact = gamma * spot
                if gamma_impact < 0.03:
                    score += 2.0
                    details.append(f"Gamma impact {gamma_impact:.4f} manageable (+2)")
                elif gamma_impact < 0.06:
                    score += 1.0
                    details.append(f"Gamma impact {gamma_impact:.4f} moderate (+1)")
                else:
                    score += 0.0
                    details.append(f"Gamma impact {gamma_impact:.4f} extreme - expiry trap risk (+0)")
            else:
                score += 1.0
                details.append("Gamma data unavailable, neutral (+1)")

            # ---- 4. Theta decay manageable (+2) ----
            if not atm_ce.empty and current_iv > 0:
                days_col_t = None
                for c in ["days_to_expiry", "dte", "days"]:
                    if c in chain.columns:
                        days_col_t = c
                        break
                dte_t = atm_ce[days_col_t].iloc[0] if days_col_t else 7
                greeks = self.calculate_greeks(spot, atm_strike, dte_t, current_iv)
                theta = greeks["theta"]

                ltp_col = None
                for c in ["ltp", "last_price", "close"]:
                    if c in atm_ce.columns:
                        ltp_col = c
                        break
                atm_premium = atm_ce[ltp_col].iloc[0] if ltp_col and not atm_ce.empty else spot * 0.02

                if atm_premium > 0:
                    theta_pct = abs(theta) / atm_premium * 100
                else:
                    theta_pct = 0

                if theta_pct < 3.0:
                    score += 2.0
                    details.append(f"Theta decay {theta_pct:.2f}% of premium/day => manageable (+2)")
                elif theta_pct < 6.0:
                    score += 1.0
                    details.append(f"Theta decay {theta_pct:.2f}% of premium/day => moderate (+1)")
                else:
                    score += 0.0
                    details.append(f"Theta decay {theta_pct:.2f}% of premium/day => heavy (+0)")
            else:
                score += 1.0
                details.append("Theta data unavailable, neutral (+1)")

            score = max(0.0, min(10.0, score))
            result["score"] = round(score, 1)
            result["bias"] = bias if 'bias' in dir() else "NEUTRAL"
            result["details"] = details

            logger.info(f"Greeks score: {result['score']}/10 | Bias: {result['bias']}")
            return result

        except Exception as e:
            logger.error(f"Greeks score() error: {e}")
            return {
                "score": 5.0,
                "bias": "NEUTRAL",
                "iv_rank": 50.0,
                "atm_delta": 0.5,
                "details": [f"Error: {e}"],
            }
