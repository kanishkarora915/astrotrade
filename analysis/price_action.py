"""
AstroNifty Price Action Analyzer
EMA, RSI, VWAP, Support/Resistance, ATR, Trend detection, composite scoring.
"""

import numpy as np
import pandas as pd
from loguru import logger


class PriceActionAnalyzer:
    """Technical price action analysis engine."""

    def __init__(self):
        self.name = "PriceActionAnalyzer"

    # ------------------------------------------------------------------ #
    #  1. Exponential Moving Average
    # ------------------------------------------------------------------ #
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.

        Parameters
        ----------
        data   : price series (typically close prices)
        period : EMA lookback period

        Returns
        -------
        pd.Series of EMA values (same length as input, NaN-padded at start).
        """
        try:
            if data is None or data.empty or period < 1:
                return pd.Series(dtype=float)
            return data.ewm(span=period, adjust=False).mean()
        except Exception as e:
            logger.error(f"calculate_ema error: {e}")
            return pd.Series(dtype=float)

    # ------------------------------------------------------------------ #
    #  2. Relative Strength Index
    # ------------------------------------------------------------------ #
    def calculate_rsi(self, data: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Wilder's smoothing).

        Returns the latest RSI value as a float 0-100.
        """
        try:
            if data is None or len(data) < period + 1:
                return 50.0

            delta = data.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)

            # Wilder's smoothed averages
            avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

            last_avg_gain = avg_gain.iloc[-1]
            last_avg_loss = avg_loss.iloc[-1]

            if last_avg_loss == 0:
                return 100.0 if last_avg_gain > 0 else 50.0

            rs = last_avg_gain / last_avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return round(rsi, 2)

        except Exception as e:
            logger.error(f"calculate_rsi error: {e}")
            return 50.0

    # ------------------------------------------------------------------ #
    #  3. VWAP (Volume Weighted Average Price)
    # ------------------------------------------------------------------ #
    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """
        Calculate VWAP from intraday or daily OHLCV data.

        Requires columns: high, low, close, volume.
        Returns the cumulative VWAP (latest value).
        """
        try:
            required = {"high", "low", "close", "volume"}
            cols_lower = {c.lower(): c for c in df.columns}

            if not required.issubset(cols_lower.keys()):
                logger.warning(f"VWAP needs {required}, got {set(cols_lower.keys())}")
                return 0.0

            high = df[cols_lower["high"]].astype(float)
            low = df[cols_lower["low"]].astype(float)
            close = df[cols_lower["close"]].astype(float)
            volume = df[cols_lower["volume"]].astype(float)

            typical_price = (high + low + close) / 3.0
            cum_tp_vol = (typical_price * volume).cumsum()
            cum_vol = volume.cumsum()

            # Avoid division by zero
            if cum_vol.iloc[-1] == 0:
                return close.iloc[-1]

            vwap = cum_tp_vol.iloc[-1] / cum_vol.iloc[-1]
            return round(vwap, 2)

        except Exception as e:
            logger.error(f"calculate_vwap error: {e}")
            return 0.0

    # ------------------------------------------------------------------ #
    #  4. Support and Resistance Detection
    # ------------------------------------------------------------------ #
    def detect_support_resistance(self, df: pd.DataFrame, window: int = 20) -> dict:
        """
        Find support and resistance levels from local minima/maxima.

        Parameters
        ----------
        df     : OHLCV DataFrame (needs 'low' and 'high' columns)
        window : rolling window to detect local extremes

        Returns
        -------
        dict with 'support' (list of levels) and 'resistance' (list of levels),
        sorted nearest-to-farthest from last close.
        """
        try:
            result = {"support": [], "resistance": []}

            cols_lower = {c.lower(): c for c in df.columns}
            if "low" not in cols_lower or "high" not in cols_lower:
                logger.warning("detect_support_resistance needs 'low' and 'high' columns")
                return result

            low = df[cols_lower["low"]].astype(float)
            high = df[cols_lower["high"]].astype(float)
            close_col = cols_lower.get("close", cols_lower.get("low"))
            last_close = df[close_col].astype(float).iloc[-1]

            # Local minima = support candidates
            supports = []
            for i in range(window, len(low) - window):
                segment = low.iloc[i - window: i + window + 1]
                if low.iloc[i] == segment.min():
                    supports.append(round(float(low.iloc[i]), 2))

            # Local maxima = resistance candidates
            resistances = []
            for i in range(window, len(high) - window):
                segment = high.iloc[i - window: i + window + 1]
                if high.iloc[i] == segment.max():
                    resistances.append(round(float(high.iloc[i]), 2))

            # Cluster nearby levels (within 0.3% of each other)
            supports = self._cluster_levels(supports, tolerance_pct=0.003)
            resistances = self._cluster_levels(resistances, tolerance_pct=0.003)

            # Only keep supports below last_close and resistances above
            supports = [s for s in supports if s < last_close]
            resistances = [r for r in resistances if r > last_close]

            # Sort by proximity to current price
            supports.sort(key=lambda x: abs(x - last_close))
            resistances.sort(key=lambda x: abs(x - last_close))

            result["support"] = supports[:5]       # top 5 nearest
            result["resistance"] = resistances[:5]

            return result

        except Exception as e:
            logger.error(f"detect_support_resistance error: {e}")
            return {"support": [], "resistance": []}

    def _cluster_levels(self, levels: list, tolerance_pct: float = 0.003) -> list:
        """Merge nearby price levels into clusters, return cluster averages."""
        if not levels:
            return []
        levels = sorted(set(levels))
        clusters = []
        current_cluster = [levels[0]]

        for i in range(1, len(levels)):
            if levels[i] <= current_cluster[-1] * (1 + tolerance_pct):
                current_cluster.append(levels[i])
            else:
                clusters.append(round(np.mean(current_cluster), 2))
                current_cluster = [levels[i]]
        clusters.append(round(np.mean(current_cluster), 2))
        return clusters

    # ------------------------------------------------------------------ #
    #  5. Average True Range
    # ------------------------------------------------------------------ #
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """
        Average True Range (Wilder's smoothing).

        Requires 'high', 'low', 'close' columns.
        Returns the latest ATR value.
        """
        try:
            cols_lower = {c.lower(): c for c in df.columns}
            required = {"high", "low", "close"}
            if not required.issubset(cols_lower.keys()):
                logger.warning(f"ATR needs {required}")
                return 0.0

            high = df[cols_lower["high"]].astype(float)
            low = df[cols_lower["low"]].astype(float)
            close = df[cols_lower["close"]].astype(float)

            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

            return round(float(atr.iloc[-1]), 2)

        except Exception as e:
            logger.error(f"calculate_atr error: {e}")
            return 0.0

    # ------------------------------------------------------------------ #
    #  6. Trend Detection via EMA crossovers
    # ------------------------------------------------------------------ #
    def detect_trend(self, df: pd.DataFrame) -> str:
        """
        Detect trend using EMA 9, 21, 50 alignment.

        Returns "BULLISH", "BEARISH", or "SIDEWAYS".
        """
        try:
            cols_lower = {c.lower(): c for c in df.columns}
            close_col = cols_lower.get("close")
            if close_col is None:
                return "SIDEWAYS"

            close = df[close_col].astype(float)

            if len(close) < 50:
                logger.debug("Not enough data for trend detection (need 50+ bars)")
                return "SIDEWAYS"

            ema9 = self.calculate_ema(close, 9)
            ema21 = self.calculate_ema(close, 21)
            ema50 = self.calculate_ema(close, 50)

            last_9 = ema9.iloc[-1]
            last_21 = ema21.iloc[-1]
            last_50 = ema50.iloc[-1]

            if last_9 > last_21 > last_50:
                return "BULLISH"
            elif last_9 < last_21 < last_50:
                return "BEARISH"
            else:
                return "SIDEWAYS"

        except Exception as e:
            logger.error(f"detect_trend error: {e}")
            return "SIDEWAYS"

    # ------------------------------------------------------------------ #
    #  7. Composite Price Action Score
    # ------------------------------------------------------------------ #
    def score(self, index: str, spot: float,
              historical_df: pd.DataFrame = None) -> dict:
        """
        Score price action out of 10.

        Scoring:
          - EMA alignment (9>21>50 bull)    : +3
          - RSI zone                        : +2
          - Above VWAP                      : +2
          - Near support: +2, near resistance: -2
          - Trend strength                  : +1

        Parameters
        ----------
        index         : instrument name (e.g. "NIFTY 50")
        spot          : current spot price
        historical_df : OHLCV DataFrame (at least 50 rows recommended)

        Returns
        -------
        dict with: score, bias, ema_trend, rsi, vwap_status, support, resistance
        """
        try:
            result = {
                "score": 5.0,
                "bias": "NEUTRAL",
                "ema_trend": "SIDEWAYS",
                "rsi": 50.0,
                "vwap_status": "UNKNOWN",
                "support": [],
                "resistance": [],
                "details": [],
            }

            if historical_df is None or historical_df.empty:
                result["details"].append("No historical data provided")
                return result

            df = historical_df.copy()
            cols_lower = {c.lower(): c for c in df.columns}
            close_col = cols_lower.get("close")
            if close_col is None:
                result["details"].append("No 'close' column found")
                return result

            close = df[close_col].astype(float)
            score_val = 0.0
            details = []

            # ---- 1. EMA alignment (+3) ----
            trend = self.detect_trend(df)
            result["ema_trend"] = trend

            if trend == "BULLISH":
                score_val += 3.0
                details.append(f"EMA 9>21>50 BULLISH alignment (+3)")
            elif trend == "BEARISH":
                score_val += 0.0
                details.append(f"EMA 9<21<50 BEARISH alignment (+0)")
            else:
                score_val += 1.5
                details.append(f"EMA mixed / SIDEWAYS (+1.5)")

            # ---- 2. RSI zone (+2) ----
            rsi = self.calculate_rsi(close, 14)
            result["rsi"] = rsi

            if 40 <= rsi <= 60:
                score_val += 2.0
                details.append(f"RSI {rsi:.1f} in neutral zone 40-60 (+2)")
            elif rsi > 60:
                if trend == "BULLISH":
                    score_val += 2.0
                    details.append(f"RSI {rsi:.1f} > 60 with bullish trend (+2)")
                else:
                    score_val += 1.0
                    details.append(f"RSI {rsi:.1f} > 60 overbought caution (+1)")
            else:  # rsi < 40
                if trend == "BEARISH":
                    score_val += 0.0
                    details.append(f"RSI {rsi:.1f} < 40 with bearish trend (+0)")
                else:
                    score_val += 1.0
                    details.append(f"RSI {rsi:.1f} < 40 oversold bounce potential (+1)")

            # ---- 3. VWAP position (+2) ----
            vwap = self.calculate_vwap(df)
            if vwap > 0:
                if spot > vwap:
                    score_val += 2.0
                    result["vwap_status"] = "ABOVE"
                    details.append(f"Spot {spot} above VWAP {vwap} (+2)")
                else:
                    score_val += 0.0
                    result["vwap_status"] = "BELOW"
                    details.append(f"Spot {spot} below VWAP {vwap} (+0)")
            else:
                score_val += 1.0
                result["vwap_status"] = "UNKNOWN"
                details.append("VWAP unavailable (+1)")

            # ---- 4. Support / Resistance proximity (+2 / -2) ----
            sr = self.detect_support_resistance(df)
            result["support"] = sr["support"]
            result["resistance"] = sr["resistance"]

            atr = self.calculate_atr(df)
            proximity_band = atr if atr > 0 else spot * 0.005

            sr_score = 0.0
            if sr["support"]:
                nearest_sup = sr["support"][0]
                if abs(spot - nearest_sup) < proximity_band:
                    sr_score += 2.0
                    details.append(f"Near support {nearest_sup} within ATR band (+2)")

            if sr["resistance"]:
                nearest_res = sr["resistance"][0]
                if abs(spot - nearest_res) < proximity_band:
                    sr_score -= 2.0
                    details.append(f"Near resistance {nearest_res} within ATR band (-2)")

            if sr_score == 0.0:
                sr_score = 1.0
                details.append("No immediate S/R proximity (+1)")

            score_val += sr_score

            # ---- 5. Trend strength bonus (+1) ----
            if len(close) >= 50:
                ema9 = self.calculate_ema(close, 9).iloc[-1]
                ema50 = self.calculate_ema(close, 50).iloc[-1]
                spread_pct = abs(ema9 - ema50) / ema50 * 100 if ema50 > 0 else 0

                if spread_pct > 1.0:
                    score_val += 1.0
                    details.append(f"EMA spread {spread_pct:.2f}% => strong trend (+1)")
                else:
                    score_val += 0.5
                    details.append(f"EMA spread {spread_pct:.2f}% => weak trend (+0.5)")
            else:
                score_val += 0.5
                details.append("Insufficient data for trend strength (+0.5)")

            # ---- Final ----
            score_val = max(0.0, min(10.0, score_val))
            result["score"] = round(score_val, 1)

            if score_val >= 7:
                result["bias"] = "BULLISH"
            elif score_val <= 3:
                result["bias"] = "BEARISH"
            else:
                result["bias"] = "NEUTRAL"

            result["details"] = details
            logger.info(f"Price Action score [{index}]: {result['score']}/10 | "
                        f"Bias: {result['bias']} | Trend: {trend}")
            return result

        except Exception as e:
            logger.error(f"Price Action score() error: {e}")
            return {
                "score": 5.0,
                "bias": "NEUTRAL",
                "ema_trend": "SIDEWAYS",
                "rsi": 50.0,
                "vwap_status": "UNKNOWN",
                "support": [],
                "resistance": [],
                "details": [f"Error: {e}"],
            }
