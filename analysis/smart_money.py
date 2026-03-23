"""
AstroNifty Smart Money Analyzer
OI sweeps, dark pool detection, institutional footprint analysis, composite scoring.
"""

import numpy as np
import pandas as pd
from loguru import logger


class SmartMoneyAnalyzer:
    """Detect institutional / smart money activity from options chain data."""

    def __init__(self):
        self.name = "SmartMoneyAnalyzer"
        # Thresholds
        self.sweep_oi_change_pct = 0.15      # 15% OI change = potential sweep
        self.sweep_min_strikes = 3           # Must span at least 3 strikes
        self.block_trade_multiplier = 3.0    # Volume > 3x avg = block trade
        self.straddle_oi_ratio = 1.5         # ATM straddle buildup threshold

    # ------------------------------------------------------------------ #
    #  1. OI Sweep Detection
    # ------------------------------------------------------------------ #
    def detect_oi_sweep(self, current_chain: pd.DataFrame,
                        prev_chain: pd.DataFrame) -> list:
        """
        Detect large OI additions across multiple strikes in the same direction.
        Institutional sweeps = big money hitting multiple strikes simultaneously.

        Parameters
        ----------
        current_chain : current option chain DataFrame
        prev_chain    : previous snapshot option chain DataFrame
                        Both need: strike, oi (or open_interest), option_type

        Returns
        -------
        List of sweep signal dicts:
          { type: 'CE_SWEEP'|'PE_SWEEP', strikes: [...], avg_oi_change_pct: float,
            direction: 'BULLISH'|'BEARISH', strength: 'HIGH'|'MEDIUM' }
        """
        try:
            sweeps = []

            if current_chain is None or prev_chain is None:
                return sweeps
            if current_chain.empty or prev_chain.empty:
                return sweeps

            # Normalise OI column name
            oi_col_curr = self._find_col(current_chain, ["oi", "open_interest", "openInterest"])
            oi_col_prev = self._find_col(prev_chain, ["oi", "open_interest", "openInterest"])
            if oi_col_curr is None or oi_col_prev is None:
                logger.warning("OI column not found in chain data")
                return sweeps

            for opt_type in ["CE", "PE"]:
                curr = current_chain.copy()
                prev = prev_chain.copy()

                type_col_curr = self._find_col(curr, ["option_type", "optionType", "type"])
                type_col_prev = self._find_col(prev, ["option_type", "optionType", "type"])

                if type_col_curr:
                    curr = curr[curr[type_col_curr].str.upper() == opt_type]
                if type_col_prev:
                    prev = prev[prev[type_col_prev].str.upper() == opt_type]

                if curr.empty or prev.empty:
                    continue

                # Merge on strike
                merged = pd.merge(
                    curr[["strike", oi_col_curr]].rename(columns={oi_col_curr: "oi_now"}),
                    prev[["strike", oi_col_prev]].rename(columns={oi_col_prev: "oi_prev"}),
                    on="strike",
                    how="inner",
                )

                merged["oi_now"] = merged["oi_now"].astype(float)
                merged["oi_prev"] = merged["oi_prev"].astype(float)
                merged["oi_change"] = merged["oi_now"] - merged["oi_prev"]
                merged["oi_change_pct"] = np.where(
                    merged["oi_prev"] > 0,
                    merged["oi_change"] / merged["oi_prev"],
                    0.0,
                )

                # Find strikes with significant OI addition
                additions = merged[merged["oi_change_pct"] > self.sweep_oi_change_pct]

                if len(additions) >= self.sweep_min_strikes:
                    avg_change = float(additions["oi_change_pct"].mean())
                    strikes_hit = sorted(additions["strike"].tolist())
                    total_oi_added = float(additions["oi_change"].sum())

                    # CE sweep = bearish (writers selling calls) or bullish (buyers accumulating)
                    # Determine by checking if OI + price increase = buying, OI + price decrease = writing
                    # Simplified: CE sweep = BEARISH (institutions writing calls)
                    #             PE sweep = BULLISH (institutions writing puts)
                    if opt_type == "CE":
                        direction = "BEARISH"
                    else:
                        direction = "BULLISH"

                    strength = "HIGH" if avg_change > 0.30 else "MEDIUM"

                    sweeps.append({
                        "type": f"{opt_type}_SWEEP",
                        "strikes": strikes_hit,
                        "num_strikes": len(strikes_hit),
                        "avg_oi_change_pct": round(avg_change * 100, 2),
                        "total_oi_added": int(total_oi_added),
                        "direction": direction,
                        "strength": strength,
                    })
                    logger.info(f"OI Sweep detected: {opt_type} across {len(strikes_hit)} strikes, "
                                f"avg change {avg_change*100:.1f}%, direction={direction}")

                # Also check for large OI unwinding (negative OI change)
                removals = merged[merged["oi_change_pct"] < -self.sweep_oi_change_pct]
                if len(removals) >= self.sweep_min_strikes:
                    avg_change = float(removals["oi_change_pct"].mean())
                    strikes_hit = sorted(removals["strike"].tolist())

                    # Unwinding CE = bullish, unwinding PE = bearish
                    if opt_type == "CE":
                        direction = "BULLISH"
                    else:
                        direction = "BEARISH"

                    sweeps.append({
                        "type": f"{opt_type}_UNWIND",
                        "strikes": strikes_hit,
                        "num_strikes": len(strikes_hit),
                        "avg_oi_change_pct": round(avg_change * 100, 2),
                        "direction": direction,
                        "strength": "HIGH" if abs(avg_change) > 0.30 else "MEDIUM",
                    })

            return sweeps

        except Exception as e:
            logger.error(f"detect_oi_sweep error: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  2. Dark Pool / Block Trade Activity
    # ------------------------------------------------------------------ #
    def detect_dark_pool_activity(self, volume_data: dict) -> dict:
        """
        Detect unusually large block trades vs historical average volume.

        Parameters
        ----------
        volume_data : dict with keys:
            - current_volume: int (today's total volume or latest candle volume)
            - avg_volume: int (20-day average volume)
            - block_trades: list of individual large trade sizes (optional)
            - total_trades: int (number of trades, optional)

        Returns
        -------
        dict with:
            - detected: bool
            - volume_ratio: float (current / avg)
            - anomaly_level: 'NONE' | 'MODERATE' | 'EXTREME'
            - block_trade_count: int
            - estimated_institutional_pct: float (0-100)
            - direction_hint: 'ACCUMULATION' | 'DISTRIBUTION' | 'UNKNOWN'
        """
        try:
            result = {
                "detected": False,
                "volume_ratio": 1.0,
                "anomaly_level": "NONE",
                "block_trade_count": 0,
                "estimated_institutional_pct": 0.0,
                "direction_hint": "UNKNOWN",
            }

            if not volume_data:
                return result

            current_vol = volume_data.get("current_volume", 0)
            avg_vol = volume_data.get("avg_volume", 0)
            block_trades = volume_data.get("block_trades", [])
            total_trades = volume_data.get("total_trades", 0)

            if avg_vol <= 0:
                return result

            # Volume ratio
            vol_ratio = current_vol / avg_vol
            result["volume_ratio"] = round(vol_ratio, 2)

            # Block trade analysis
            if block_trades:
                avg_trade_size = current_vol / total_trades if total_trades > 0 else current_vol
                large_trades = [t for t in block_trades
                                if t > avg_trade_size * self.block_trade_multiplier]
                result["block_trade_count"] = len(large_trades)

                if large_trades:
                    block_volume = sum(large_trades)
                    result["estimated_institutional_pct"] = round(
                        min(100.0, (block_volume / current_vol) * 100), 1
                    ) if current_vol > 0 else 0.0

            # Anomaly classification
            if vol_ratio >= 4.0:
                result["anomaly_level"] = "EXTREME"
                result["detected"] = True
            elif vol_ratio >= 2.0:
                result["anomaly_level"] = "MODERATE"
                result["detected"] = True
            elif result["block_trade_count"] >= 5:
                result["anomaly_level"] = "MODERATE"
                result["detected"] = True

            # Direction hint from volume pattern
            price_change = volume_data.get("price_change", None)
            if price_change is not None:
                if price_change > 0 and vol_ratio > 2.0:
                    result["direction_hint"] = "ACCUMULATION"
                elif price_change < 0 and vol_ratio > 2.0:
                    result["direction_hint"] = "DISTRIBUTION"

            if result["detected"]:
                logger.info(f"Dark pool activity: ratio={vol_ratio:.1f}x, "
                            f"level={result['anomaly_level']}, "
                            f"blocks={result['block_trade_count']}")

            return result

        except Exception as e:
            logger.error(f"detect_dark_pool_activity error: {e}")
            return {
                "detected": False,
                "volume_ratio": 1.0,
                "anomaly_level": "NONE",
                "block_trade_count": 0,
                "estimated_institutional_pct": 0.0,
                "direction_hint": "UNKNOWN",
            }

    # ------------------------------------------------------------------ #
    #  3. Institutional Footprint Detection
    # ------------------------------------------------------------------ #
    def detect_institutional_footprint(self, chain: pd.DataFrame,
                                       prev_chain: pd.DataFrame,
                                       spot: float) -> dict:
        """
        Check if institutions are quietly building positions.
        Looks for ATM straddle/strangle buildup patterns.

        Parameters
        ----------
        chain      : current option chain DataFrame
        prev_chain : previous option chain DataFrame
        spot       : current spot price

        Returns
        -------
        dict with:
            - footprint_detected: bool
            - pattern: 'STRADDLE_BUILDUP' | 'STRANGLE_BUILDUP' | 'DIRECTIONAL' | 'NONE'
            - bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
            - atm_ce_oi_change: int
            - atm_pe_oi_change: int
            - pcr_shift: float (change in put-call ratio)
            - details: list of str
        """
        try:
            result = {
                "footprint_detected": False,
                "pattern": "NONE",
                "bias": "NEUTRAL",
                "atm_ce_oi_change": 0,
                "atm_pe_oi_change": 0,
                "pcr_shift": 0.0,
                "details": [],
            }

            if chain is None or prev_chain is None or chain.empty or prev_chain.empty:
                result["details"].append("Insufficient chain data")
                return result

            oi_col = self._find_col(chain, ["oi", "open_interest", "openInterest"])
            type_col = self._find_col(chain, ["option_type", "optionType", "type"])
            if oi_col is None or type_col is None:
                result["details"].append("Required columns missing")
                return result

            oi_col_prev = self._find_col(prev_chain, ["oi", "open_interest", "openInterest"])
            type_col_prev = self._find_col(prev_chain, ["option_type", "optionType", "type"])
            if oi_col_prev is None or type_col_prev is None:
                result["details"].append("Required columns missing in prev_chain")
                return result

            # Find ATM strike
            strikes = chain["strike"].unique()
            atm_strike = min(strikes, key=lambda s: abs(s - spot))

            # Near-ATM range (ATM +/- 1 strike step)
            sorted_strikes = sorted(strikes)
            atm_idx = list(sorted_strikes).index(atm_strike)
            near_strikes = sorted_strikes[max(0, atm_idx - 1): atm_idx + 2]

            # Calculate OI changes for CE and PE at near-ATM strikes
            ce_oi_change = 0
            pe_oi_change = 0

            for strike in near_strikes:
                for opt in ["CE", "PE"]:
                    curr_rows = chain[(chain["strike"] == strike) &
                                      (chain[type_col].str.upper() == opt)]
                    prev_rows = prev_chain[(prev_chain["strike"] == strike) &
                                            (prev_chain[type_col_prev].str.upper() == opt)]

                    if not curr_rows.empty and not prev_rows.empty:
                        curr_oi = float(curr_rows[oi_col].iloc[0])
                        prev_oi = float(prev_rows[oi_col_prev].iloc[0])
                        change = curr_oi - prev_oi

                        if opt == "CE":
                            ce_oi_change += change
                        else:
                            pe_oi_change += change

            result["atm_ce_oi_change"] = int(ce_oi_change)
            result["atm_pe_oi_change"] = int(pe_oi_change)

            # PCR shift
            total_ce_oi_curr = float(chain[chain[type_col].str.upper() == "CE"][oi_col].sum())
            total_pe_oi_curr = float(chain[chain[type_col].str.upper() == "PE"][oi_col].sum())
            total_ce_oi_prev = float(prev_chain[prev_chain[type_col_prev].str.upper() == "CE"][oi_col_prev].sum())
            total_pe_oi_prev = float(prev_chain[prev_chain[type_col_prev].str.upper() == "PE"][oi_col_prev].sum())

            pcr_curr = total_pe_oi_curr / total_ce_oi_curr if total_ce_oi_curr > 0 else 1.0
            pcr_prev = total_pe_oi_prev / total_ce_oi_prev if total_ce_oi_prev > 0 else 1.0
            pcr_shift = pcr_curr - pcr_prev
            result["pcr_shift"] = round(pcr_shift, 4)

            # ---- Pattern Detection ----
            # Straddle buildup: both CE and PE OI increasing significantly at ATM
            avg_atm_oi = (total_ce_oi_curr + total_pe_oi_curr) / (2 * max(len(strikes), 1))
            ce_change_significant = abs(ce_oi_change) > avg_atm_oi * 0.1
            pe_change_significant = abs(pe_oi_change) > avg_atm_oi * 0.1

            if ce_oi_change > 0 and pe_oi_change > 0 and ce_change_significant and pe_change_significant:
                # Both sides building = straddle
                ratio = max(ce_oi_change, pe_oi_change) / max(min(ce_oi_change, pe_oi_change), 1)
                if ratio < self.straddle_oi_ratio:
                    result["footprint_detected"] = True
                    result["pattern"] = "STRADDLE_BUILDUP"
                    result["bias"] = "NEUTRAL"
                    result["details"].append(
                        f"ATM straddle buildup: CE OI +{int(ce_oi_change)}, PE OI +{int(pe_oi_change)}"
                    )
                else:
                    result["footprint_detected"] = True
                    result["pattern"] = "STRANGLE_BUILDUP"
                    result["bias"] = "NEUTRAL"
                    result["details"].append(
                        f"Near-ATM strangle buildup: CE/PE ratio {ratio:.2f}"
                    )

            elif ce_change_significant and not pe_change_significant:
                # Only CE OI building = bearish (call writing)
                if ce_oi_change > 0:
                    result["footprint_detected"] = True
                    result["pattern"] = "DIRECTIONAL"
                    result["bias"] = "BEARISH"
                    result["details"].append(
                        f"Heavy CE writing at ATM: +{int(ce_oi_change)} OI (bearish)"
                    )

            elif pe_change_significant and not ce_change_significant:
                # Only PE OI building = bullish (put writing)
                if pe_oi_change > 0:
                    result["footprint_detected"] = True
                    result["pattern"] = "DIRECTIONAL"
                    result["bias"] = "BULLISH"
                    result["details"].append(
                        f"Heavy PE writing at ATM: +{int(pe_oi_change)} OI (bullish)"
                    )

            # PCR shift insight
            if abs(pcr_shift) > 0.05:
                if pcr_shift > 0:
                    result["details"].append(
                        f"PCR shifted up by {pcr_shift:.3f} (more puts = support building)"
                    )
                else:
                    result["details"].append(
                        f"PCR shifted down by {abs(pcr_shift):.3f} (more calls = resistance building)"
                    )

            if result["footprint_detected"]:
                logger.info(f"Institutional footprint: {result['pattern']}, bias={result['bias']}")

            return result

        except Exception as e:
            logger.error(f"detect_institutional_footprint error: {e}")
            return {
                "footprint_detected": False,
                "pattern": "NONE",
                "bias": "NEUTRAL",
                "atm_ce_oi_change": 0,
                "atm_pe_oi_change": 0,
                "pcr_shift": 0.0,
                "details": [f"Error: {e}"],
            }

    # ------------------------------------------------------------------ #
    #  4. Composite Smart Money Score
    # ------------------------------------------------------------------ #
    def score(self, chain: pd.DataFrame, prev_chain: pd.DataFrame,
              spot: float = None) -> dict:
        """
        Smart money composite score out of 5.

        Scoring:
          - No sweep detected (clean market)     : +2
          - Smart money aligned with bias         : +2
          - No dark pool anomaly                  : +1

        Parameters
        ----------
        chain      : current option chain
        prev_chain : previous option chain
        spot       : current spot price (optional, derived from chain if absent)

        Returns
        -------
        dict with: score, bias, signals (list), details (list)
        """
        try:
            result = {
                "score": 2.5,
                "bias": "NEUTRAL",
                "signals": [],
                "details": [],
            }

            if chain is None or chain.empty:
                result["details"].append("No chain data")
                return result
            if prev_chain is None or prev_chain.empty:
                result["details"].append("No previous chain data")
                return result

            # Derive spot from chain if not provided
            if spot is None or spot == 0:
                if "spot" in chain.columns:
                    spot = float(chain["spot"].iloc[0])
                elif "strike" in chain.columns:
                    ltp_col = self._find_col(chain, ["ltp", "last_price", "close"])
                    if ltp_col:
                        # Approximate spot from ATM strike
                        spot = float(chain["strike"].median())
                    else:
                        spot = float(chain["strike"].median())

            score_val = 0.0
            details = []
            signals = []

            # ---- 1. OI Sweep check (+2 if clean) ----
            sweeps = self.detect_oi_sweep(chain, prev_chain)

            if not sweeps:
                score_val += 2.0
                details.append("No OI sweep detected - clean market (+2)")
            else:
                # Sweeps present - check alignment
                sweep_directions = [s["direction"] for s in sweeps]
                signals.extend(sweeps)

                bullish_sweeps = sweep_directions.count("BULLISH")
                bearish_sweeps = sweep_directions.count("BEARISH")

                if bullish_sweeps > 0 and bearish_sweeps == 0:
                    score_val += 1.0
                    details.append(f"Bullish OI sweep(s) detected ({bullish_sweeps}) - directional (+1)")
                elif bearish_sweeps > 0 and bullish_sweeps == 0:
                    score_val += 1.0
                    details.append(f"Bearish OI sweep(s) detected ({bearish_sweeps}) - directional (+1)")
                else:
                    score_val += 0.5
                    details.append(f"Mixed OI sweeps (bull={bullish_sweeps}, bear={bearish_sweeps}) (+0.5)")

            # ---- 2. Institutional footprint alignment (+2) ----
            footprint = self.detect_institutional_footprint(chain, prev_chain, spot)

            if footprint["footprint_detected"]:
                signals.append({
                    "type": "INSTITUTIONAL_FOOTPRINT",
                    "pattern": footprint["pattern"],
                    "bias": footprint["bias"],
                })

                if footprint["pattern"] in ["STRADDLE_BUILDUP", "STRANGLE_BUILDUP"]:
                    # Neutral institutional activity - range-bound expectation
                    score_val += 1.5
                    details.append(f"Institutional {footprint['pattern']} = range-bound expectation (+1.5)")
                elif footprint["pattern"] == "DIRECTIONAL":
                    score_val += 2.0
                    details.append(
                        f"Directional institutional footprint: {footprint['bias']} (+2)"
                    )
                else:
                    score_val += 1.0
                    details.append("Institutional activity unclear (+1)")
            else:
                score_val += 1.0
                details.append("No institutional footprint detected (+1)")

            # ---- 3. Dark pool / volume anomaly (+1 if clean) ----
            # Build volume_data from chain if possible
            vol_col = self._find_col(chain, ["volume", "total_volume", "tradedVolume"])
            vol_col_prev = self._find_col(prev_chain, ["volume", "total_volume", "tradedVolume"])

            if vol_col and vol_col_prev:
                current_volume = int(chain[vol_col].sum())
                avg_volume = int(prev_chain[vol_col_prev].sum())

                volume_data = {
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,
                }

                dark_pool = self.detect_dark_pool_activity(volume_data)

                if not dark_pool["detected"]:
                    score_val += 1.0
                    details.append(f"No dark pool anomaly (vol ratio {dark_pool['volume_ratio']:.1f}x) (+1)")
                else:
                    score_val += 0.0
                    signals.append({
                        "type": "DARK_POOL",
                        "anomaly_level": dark_pool["anomaly_level"],
                        "volume_ratio": dark_pool["volume_ratio"],
                    })
                    details.append(
                        f"Dark pool activity: {dark_pool['anomaly_level']} "
                        f"(vol {dark_pool['volume_ratio']:.1f}x avg) (+0)"
                    )
            else:
                score_val += 0.5
                details.append("Volume data unavailable for dark pool check (+0.5)")

            # ---- Determine overall bias ----
            bias_votes = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
            for sig in signals:
                b = sig.get("bias") or sig.get("direction", "NEUTRAL")
                if b in bias_votes:
                    bias_votes[b] += 1

            if footprint.get("bias") and footprint["bias"] != "NEUTRAL":
                bias_votes[footprint["bias"]] += 1

            if bias_votes["BULLISH"] > bias_votes["BEARISH"]:
                overall_bias = "BULLISH"
            elif bias_votes["BEARISH"] > bias_votes["BULLISH"]:
                overall_bias = "BEARISH"
            else:
                overall_bias = "NEUTRAL"

            score_val = max(0.0, min(5.0, score_val))
            result["score"] = round(score_val, 1)
            result["bias"] = overall_bias
            result["signals"] = signals
            result["details"] = details

            logger.info(f"Smart Money score: {result['score']}/5 | Bias: {result['bias']} | "
                        f"Signals: {len(signals)}")
            return result

        except Exception as e:
            logger.error(f"Smart Money score() error: {e}")
            return {
                "score": 2.5,
                "bias": "NEUTRAL",
                "signals": [],
                "details": [f"Error: {e}"],
            }

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #
    def _find_col(self, df: pd.DataFrame, candidates: list) -> str | None:
        """Return the first matching column name from candidates."""
        if df is None:
            return None
        for c in candidates:
            if c in df.columns:
                return c
        # Case-insensitive fallback
        lower_map = {col.lower(): col for col in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None
