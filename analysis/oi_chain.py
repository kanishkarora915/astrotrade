"""
OI Chain Analysis Engine
Complete Open Interest analysis: max pain, OI walls, PCR, buildup patterns,
spike detection, gamma exposure, and composite scoring.
"""

import numpy as np
import pandas as pd
from loguru import logger
from typing import Dict, List, Optional


class OIChainAnalyzer:
    """
    Analyzes option chain OI data to derive actionable trading signals.

    Expected DataFrame columns:
        strike, ce_oi, ce_volume, ce_ltp, ce_iv,
        pe_oi, pe_volume, pe_ltp, pe_iv
    """

    NIFTY_LOT_SIZE = 25

    # ------------------------------------------------------------------ #
    #  1. Max Pain                                                        #
    # ------------------------------------------------------------------ #
    def calculate_max_pain(self, chain: pd.DataFrame) -> int:
        """
        For each candidate expiry strike, compute the total option-writer
        loss across all strikes.  Return the strike that minimises that loss.

        CE loss at expiry E for a given strike K with OI:
            max(0, K - E) * CE_OI   (writers pay intrinsic to CE holders)
        PE loss at expiry E for a given strike K with OI:
            max(0, E - K) * PE_OI   (writers pay intrinsic to PE holders)
        """
        try:
            df = chain.copy()
            strikes = df["strike"].values
            ce_oi = df["ce_oi"].fillna(0).values
            pe_oi = df["pe_oi"].fillna(0).values

            min_loss = np.inf
            max_pain_strike = int(strikes[0])

            for expiry in strikes:
                ce_loss = np.sum(np.maximum(strikes - expiry, 0) * ce_oi)
                pe_loss = np.sum(np.maximum(expiry - strikes, 0) * pe_oi)
                total = ce_loss + pe_loss
                if total < min_loss:
                    min_loss = total
                    max_pain_strike = int(expiry)

            logger.debug(f"Max pain calculated: {max_pain_strike} (loss={min_loss:,.0f})")
            return max_pain_strike

        except Exception as e:
            logger.error(f"calculate_max_pain failed: {e}")
            return 0

    # ------------------------------------------------------------------ #
    #  2. OI Walls                                                        #
    # ------------------------------------------------------------------ #
    def detect_oi_walls(self, chain: pd.DataFrame) -> dict:
        """
        Identify primary & secondary support/resistance walls from OI
        concentration.  A wall is 'significant' when its OI exceeds
        1.5x the mean OI for that option type.
        """
        try:
            df = chain.copy()
            df["ce_oi"] = df["ce_oi"].fillna(0)
            df["pe_oi"] = df["pe_oi"].fillna(0)

            # Primary walls
            ce_wall_idx = df["ce_oi"].idxmax()
            pe_wall_idx = df["pe_oi"].idxmax()

            ce_wall = int(df.loc[ce_wall_idx, "strike"])
            ce_oi_val = int(df.loc[ce_wall_idx, "ce_oi"])
            pe_wall = int(df.loc[pe_wall_idx, "strike"])
            pe_oi_val = int(df.loc[pe_wall_idx, "pe_oi"])

            # Significance threshold
            ce_avg = df["ce_oi"].mean()
            pe_avg = df["pe_oi"].mean()
            ce_threshold = ce_avg * 1.5
            pe_threshold = pe_avg * 1.5

            # Secondary walls (significant but not the primary)
            secondary_ce = (
                df.loc[(df["ce_oi"] > ce_threshold) & (df["strike"] != ce_wall)]
                .nlargest(3, "ce_oi")[["strike", "ce_oi"]]
                .apply(lambda r: {"strike": int(r["strike"]), "oi": int(r["ce_oi"])}, axis=1)
                .tolist()
            )

            secondary_pe = (
                df.loc[(df["pe_oi"] > pe_threshold) & (df["strike"] != pe_wall)]
                .nlargest(3, "pe_oi")[["strike", "pe_oi"]]
                .apply(lambda r: {"strike": int(r["strike"]), "oi": int(r["pe_oi"])}, axis=1)
                .tolist()
            )

            result = {
                "ce_wall": ce_wall,
                "ce_oi": ce_oi_val,
                "ce_significant": ce_oi_val > ce_threshold,
                "pe_wall": pe_wall,
                "pe_oi": pe_oi_val,
                "pe_significant": pe_oi_val > pe_threshold,
                "secondary_ce": secondary_ce,
                "secondary_pe": secondary_pe,
            }
            logger.debug(f"OI walls: CE {ce_wall} ({ce_oi_val:,}) | PE {pe_wall} ({pe_oi_val:,})")
            return result

        except Exception as e:
            logger.error(f"detect_oi_walls failed: {e}")
            return {
                "ce_wall": 0, "ce_oi": 0, "pe_wall": 0, "pe_oi": 0,
                "ce_significant": False, "pe_significant": False,
                "secondary_ce": [], "secondary_pe": [],
            }

    # ------------------------------------------------------------------ #
    #  3. Put-Call Ratio                                                   #
    # ------------------------------------------------------------------ #
    def calculate_pcr(self, chain: pd.DataFrame) -> dict:
        """
        Overall PCR  = Total PE OI / Total CE OI
        Weighted PCR = sum(PE_OI * strike) / sum(CE_OI * strike)
        Interpretation: >1.3 bullish, 0.8-1.3 neutral, <0.8 bearish.
        """
        try:
            df = chain.copy()
            total_ce = df["ce_oi"].fillna(0).sum()
            total_pe = df["pe_oi"].fillna(0).sum()

            pcr_overall = round(total_pe / total_ce, 4) if total_ce > 0 else 0.0

            weighted_pe = (df["pe_oi"].fillna(0) * df["strike"]).sum()
            weighted_ce = (df["ce_oi"].fillna(0) * df["strike"]).sum()
            pcr_weighted = round(weighted_pe / weighted_ce, 4) if weighted_ce > 0 else 0.0

            if pcr_overall > 1.3:
                interpretation = "BULLISH"
                score_pts = 5
            elif pcr_overall < 0.8:
                interpretation = "BEARISH"
                score_pts = -5
            else:
                interpretation = "NEUTRAL"
                score_pts = 0

            result = {
                "pcr_overall": pcr_overall,
                "pcr_weighted": pcr_weighted,
                "total_ce_oi": int(total_ce),
                "total_pe_oi": int(total_pe),
                "interpretation": interpretation,
                "score_pts": score_pts,
            }
            logger.debug(f"PCR: {pcr_overall:.2f} ({interpretation})")
            return result

        except Exception as e:
            logger.error(f"calculate_pcr failed: {e}")
            return {
                "pcr_overall": 0.0, "pcr_weighted": 0.0,
                "total_ce_oi": 0, "total_pe_oi": 0,
                "interpretation": "UNKNOWN", "score_pts": 0,
            }

    # ------------------------------------------------------------------ #
    #  4. OI Buildup Pattern                                              #
    # ------------------------------------------------------------------ #
    def detect_oi_buildup_pattern(
        self,
        current: pd.DataFrame,
        prev: pd.DataFrame,
        spot_price: float,
    ) -> dict:
        """
        Compare current vs previous chain around ATM strikes (spot +/- 2
        strikes) to classify the buildup pattern.

        Price UP  + OI UP   = Fresh Long Buildup  (+10)
        Price UP  + OI DOWN = Short Covering       (+4)
        Price DOWN + OI UP  = Fresh Short Buildup  (-10)
        Price DOWN + OI DOWN = Long Unwinding      (-4)
        """
        try:
            cur = current.copy().sort_values("strike").reset_index(drop=True)
            prv = prev.copy().sort_values("strike").reset_index(drop=True)

            # Find ATM index (closest strike to spot)
            atm_idx = (cur["strike"] - spot_price).abs().idxmin()
            lo = max(0, atm_idx - 2)
            hi = min(len(cur) - 1, atm_idx + 2)
            atm_strikes = cur.loc[lo:hi, "strike"].values

            # Aggregate OI and price around ATM
            cur_atm = cur[cur["strike"].isin(atm_strikes)]
            prv_atm = prv[prv["strike"].isin(atm_strikes)]

            if cur_atm.empty or prv_atm.empty:
                return {"pattern": "NO_DATA", "score_pts": 0, "details": {}}

            # Use CE LTP change as proxy for underlying price move
            cur_ce_ltp = cur_atm["ce_ltp"].fillna(0).mean()
            prv_ce_ltp = prv_atm["ce_ltp"].fillna(0).mean()
            price_change = cur_ce_ltp - prv_ce_ltp  # positive = spot went up

            # Total OI change (both CE and PE combined for ATM strikes)
            cur_total_oi = cur_atm["ce_oi"].fillna(0).sum() + cur_atm["pe_oi"].fillna(0).sum()
            prv_total_oi = prv_atm["ce_oi"].fillna(0).sum() + prv_atm["pe_oi"].fillna(0).sum()
            oi_change = cur_total_oi - prv_total_oi

            price_up = price_change > 0
            oi_up = oi_change > 0

            if price_up and oi_up:
                pattern = "FRESH_LONG_BUILDUP"
                score_pts = 10
            elif price_up and not oi_up:
                pattern = "SHORT_COVERING"
                score_pts = 4
            elif not price_up and oi_up:
                pattern = "FRESH_SHORT_BUILDUP"
                score_pts = -10
            else:
                pattern = "LONG_UNWINDING"
                score_pts = -4

            details = {
                "atm_strikes": [int(s) for s in atm_strikes],
                "price_change_proxy": round(price_change, 2),
                "oi_change": int(oi_change),
                "cur_total_oi": int(cur_total_oi),
                "prv_total_oi": int(prv_total_oi),
            }

            logger.debug(f"Buildup: {pattern} (price_chg={price_change:.1f}, oi_chg={oi_change:,})")
            return {"pattern": pattern, "score_pts": score_pts, "details": details}

        except Exception as e:
            logger.error(f"detect_oi_buildup_pattern failed: {e}")
            return {"pattern": "ERROR", "score_pts": 0, "details": {"error": str(e)}}

    # ------------------------------------------------------------------ #
    #  5. OI Spike Detection (Smart Money)                                #
    # ------------------------------------------------------------------ #
    def detect_oi_spike(
        self,
        current: pd.DataFrame,
        prev: pd.DataFrame,
    ) -> list:
        """
        Detect strikes where OI surged > 2x the normal change but the
        option price barely moved (< 5 pts).  These are potential smart-
        money accumulation zones.
        """
        try:
            cur = current.copy().sort_values("strike").reset_index(drop=True)
            prv = prev.copy().sort_values("strike").reset_index(drop=True)

            merged = cur.merge(prv, on="strike", suffixes=("_cur", "_prv"))
            suspicious: List[dict] = []

            for side in ("ce", "pe"):
                oi_cur_col = f"{side}_oi_cur"
                oi_prv_col = f"{side}_oi_prv"
                ltp_cur_col = f"{side}_ltp_cur"
                ltp_prv_col = f"{side}_ltp_prv"

                merged[f"{side}_oi_chg"] = (
                    merged[oi_cur_col].fillna(0) - merged[oi_prv_col].fillna(0)
                )
                merged[f"{side}_price_chg"] = (
                    (merged[ltp_cur_col].fillna(0) - merged[ltp_prv_col].fillna(0)).abs()
                )

            for side in ("ce", "pe"):
                oi_chg_col = f"{side}_oi_chg"
                price_chg_col = f"{side}_price_chg"

                avg_oi_chg = merged[oi_chg_col].abs().mean()
                threshold = avg_oi_chg * 2 if avg_oi_chg > 0 else 1

                spike_mask = (merged[oi_chg_col].abs() > threshold) & (
                    merged[price_chg_col] < 5
                )
                spikes = merged.loc[spike_mask]

                for _, row in spikes.iterrows():
                    suspicious.append({
                        "strike": int(row["strike"]),
                        "side": side.upper(),
                        "oi_change": int(row[oi_chg_col]),
                        "price_change": round(row[price_chg_col], 2),
                        "signal": "SMART_MONEY_ACCUMULATION",
                    })

            logger.debug(f"OI spikes detected: {len(suspicious)} suspicious strikes")
            return suspicious

        except Exception as e:
            logger.error(f"detect_oi_spike failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    #  6. Gamma Exposure (GEX)                                            #
    # ------------------------------------------------------------------ #
    def calculate_gamma_exposure(
        self,
        chain: pd.DataFrame,
        spot: float,
    ) -> dict:
        """
        Approximate Gamma Exposure:
            GEX_per_strike = Gamma * OI * lot_size * spot^2 * 0.01
        CE gamma is positive for dealers, PE gamma is negative.

        Positive net GEX  -> dealers long gamma -> range-bound / pinning.
        Negative net GEX  -> dealers short gamma -> trending / volatile.
        """
        try:
            df = chain.copy()
            lot = self.NIFTY_LOT_SIZE
            spot2 = spot * spot * 0.01

            # Estimate gamma from IV and time (Black-Scholes approximation)
            # gamma ~ N'(d1) / (S * sigma * sqrt(T))
            # We approximate using IV and assume ~5 trading days to expiry
            T = 5 / 252  # default assumption
            sqrt_T = np.sqrt(T) if T > 0 else 0.001

            def _approx_gamma(iv: float) -> float:
                """Simple gamma proxy from IV."""
                sigma = iv / 100 if iv > 0 else 0.15
                denom = spot * sigma * sqrt_T
                if denom == 0:
                    return 0.0
                # N'(0) ~ 0.3989 (ATM gamma is highest)
                return 0.3989 / denom

            df["ce_gamma"] = df["ce_iv"].fillna(15).apply(_approx_gamma)
            df["pe_gamma"] = df["pe_iv"].fillna(15).apply(_approx_gamma)

            df["ce_gex"] = df["ce_gamma"] * df["ce_oi"].fillna(0) * lot * spot2
            df["pe_gex"] = -(df["pe_gamma"] * df["pe_oi"].fillna(0) * lot * spot2)

            total_ce_gex = df["ce_gex"].sum()
            total_pe_gex = df["pe_gex"].sum()
            net_gex = total_ce_gex + total_pe_gex

            if net_gex > 0:
                interpretation = "POSITIVE_GAMMA_RANGE_BOUND"
                expected_move = "LOW"
            else:
                interpretation = "NEGATIVE_GAMMA_TRENDING"
                expected_move = "HIGH"

            # Estimate expected daily range from GEX magnitude
            gex_magnitude = abs(net_gex)
            # Normalise to a rough points estimate
            expected_pts = round(spot * 0.01 * (1 - min(gex_magnitude / 1e9, 0.5)), 1)

            result = {
                "net_gex": round(net_gex, 2),
                "ce_gex": round(total_ce_gex, 2),
                "pe_gex": round(total_pe_gex, 2),
                "interpretation": interpretation,
                "expected_move": expected_move,
                "expected_range_pts": expected_pts,
            }
            logger.debug(
                f"GEX: net={net_gex:,.0f} | {interpretation} | ~{expected_pts} pts range"
            )
            return result

        except Exception as e:
            logger.error(f"calculate_gamma_exposure failed: {e}")
            return {
                "net_gex": 0.0, "ce_gex": 0.0, "pe_gex": 0.0,
                "interpretation": "UNKNOWN", "expected_move": "UNKNOWN",
                "expected_range_pts": 0.0,
            }

    # ------------------------------------------------------------------ #
    #  7. Composite Score (/25)                                           #
    # ------------------------------------------------------------------ #
    def score(
        self,
        chain: pd.DataFrame,
        prev_chain: pd.DataFrame,
        spot: float,
    ) -> dict:
        """
        Combined OI-based directional score out of 25.

        Breakdown:
            Max pain within 100 pts of spot  : +5
            PCR bullish                      : +5  (bearish: -5)
            Fresh long buildup               : +8  (short: -8)
            OI walls clearly defined         : +4
            No unusual OI spike              : +3
        """
        try:
            total_score = 0
            breakdown: Dict[str, int] = {}

            # --- Max Pain ---
            max_pain = self.calculate_max_pain(chain)
            mp_distance = abs(max_pain - spot)
            if mp_distance <= 100:
                mp_pts = 5
            elif mp_distance <= 200:
                mp_pts = 3
            else:
                mp_pts = 0
            total_score += mp_pts
            breakdown["max_pain"] = mp_pts

            # --- PCR ---
            pcr_data = self.calculate_pcr(chain)
            pcr_pts = pcr_data["score_pts"]
            total_score += pcr_pts
            breakdown["pcr"] = pcr_pts

            # --- Buildup ---
            buildup = self.detect_oi_buildup_pattern(chain, prev_chain, spot)
            pattern = buildup["pattern"]
            if pattern == "FRESH_LONG_BUILDUP":
                bu_pts = 8
            elif pattern == "SHORT_COVERING":
                bu_pts = 4
            elif pattern == "FRESH_SHORT_BUILDUP":
                bu_pts = -8
            elif pattern == "LONG_UNWINDING":
                bu_pts = -4
            else:
                bu_pts = 0
            total_score += bu_pts
            breakdown["buildup"] = bu_pts

            # --- OI Walls ---
            walls = self.detect_oi_walls(chain)
            walls_clear = walls["ce_significant"] and walls["pe_significant"]
            wall_pts = 4 if walls_clear else 2
            total_score += wall_pts
            breakdown["walls"] = wall_pts

            # --- OI Spike ---
            spikes = self.detect_oi_spike(chain, prev_chain)
            spike_pts = 3 if len(spikes) == 0 else 0
            total_score += spike_pts
            breakdown["spike_clean"] = spike_pts

            # Clamp to [-25, 25]
            total_score = max(-25, min(25, total_score))

            if total_score >= 10:
                bias = "BULLISH"
            elif total_score <= -10:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            result = {
                "score": total_score,
                "max_score": 25,
                "bias": bias,
                "breakdown": breakdown,
                "max_pain": max_pain,
                "max_pain_distance": round(mp_distance, 1),
                "pcr": pcr_data,
                "walls": walls,
                "buildup_pattern": pattern,
                "oi_spikes": len(spikes),
            }
            logger.info(f"OI Score: {total_score}/25 | Bias: {bias}")
            return result

        except Exception as e:
            logger.error(f"score failed: {e}")
            return {
                "score": 0, "max_score": 25, "bias": "UNKNOWN",
                "breakdown": {}, "max_pain": 0, "pcr": {},
                "walls": {}, "buildup_pattern": "ERROR", "oi_spikes": 0,
            }

    # ------------------------------------------------------------------ #
    #  8. Buildup Score (/22)                                             #
    # ------------------------------------------------------------------ #
    def buildup_score(
        self,
        chain: pd.DataFrame,
        prev_chain: pd.DataFrame,
        spot: float,
    ) -> dict:
        """
        Deeper OI buildup analysis scored out of 22.

        Breakdown:
            ATM buildup pattern           : up to +/-10
            ITM CE OI unwinding (bullish) : +4 / building: -4
            OTM PE OI building (bullish)  : +4 / unwinding: -4
            Volume confirmation           : +4
        """
        try:
            total_score = 0
            breakdown: Dict[str, int] = {}

            cur = chain.copy().sort_values("strike").reset_index(drop=True)
            prv = prev_chain.copy().sort_values("strike").reset_index(drop=True)

            atm_idx = (cur["strike"] - spot).abs().idxmin()
            atm_strike = cur.loc[atm_idx, "strike"]

            # --- ATM buildup (reuse method 4) ---
            buildup = self.detect_oi_buildup_pattern(chain, prev_chain, spot)
            atm_pts = buildup["score_pts"]
            total_score += atm_pts
            breakdown["atm_buildup"] = atm_pts

            # --- ITM CE analysis ---
            # ITM CE = strikes below spot.  If CE OI is decreasing, writers
            # are closing  -> bullish.
            itm_ce_cur = cur.loc[cur["strike"] < atm_strike, "ce_oi"].fillna(0).sum()
            itm_ce_prv = prv.loc[prv["strike"] < atm_strike, "ce_oi"].fillna(0).sum()
            itm_ce_chg = itm_ce_cur - itm_ce_prv

            if itm_ce_chg < 0:
                itm_pts = 4  # unwinding = bullish
            elif itm_ce_chg > 0:
                itm_pts = -4  # building = bearish
            else:
                itm_pts = 0
            total_score += itm_pts
            breakdown["itm_ce"] = itm_pts

            # --- OTM PE analysis ---
            # OTM PE = strikes below spot.  If PE OI is increasing, writers
            # are selling PEs  -> bullish.
            otm_pe_cur = cur.loc[cur["strike"] < atm_strike, "pe_oi"].fillna(0).sum()
            otm_pe_prv = prv.loc[prv["strike"] < atm_strike, "pe_oi"].fillna(0).sum()
            otm_pe_chg = otm_pe_cur - otm_pe_prv

            if otm_pe_chg > 0:
                pe_pts = 4  # PE OI building below spot = bullish (writers selling puts)
            elif otm_pe_chg < 0:
                pe_pts = -4  # PE OI unwinding = bearish
            else:
                pe_pts = 0
            total_score += pe_pts
            breakdown["otm_pe"] = pe_pts

            # --- Volume confirmation ---
            # If CE volume near ATM > PE volume -> bullish activity
            lo = max(0, atm_idx - 2)
            hi = min(len(cur) - 1, atm_idx + 2)
            atm_range = cur.loc[lo:hi]

            ce_vol = atm_range["ce_volume"].fillna(0).sum()
            pe_vol = atm_range["pe_volume"].fillna(0).sum()

            if ce_vol > 0 and pe_vol > 0:
                vol_ratio = ce_vol / pe_vol
                if vol_ratio > 1.3:
                    vol_pts = 4  # More CE buying = bullish
                elif vol_ratio < 0.7:
                    vol_pts = -4  # More PE buying = bearish
                else:
                    vol_pts = 0
            else:
                vol_pts = 0
            total_score += vol_pts
            breakdown["volume_confirm"] = vol_pts

            # Clamp
            total_score = max(-22, min(22, total_score))

            if total_score >= 8:
                bias = "BULLISH"
            elif total_score <= -8:
                bias = "BEARISH"
            else:
                bias = "NEUTRAL"

            result = {
                "score": total_score,
                "max_score": 22,
                "bias": bias,
                "breakdown": breakdown,
                "details": {
                    "itm_ce_change": int(itm_ce_chg),
                    "otm_pe_change": int(otm_pe_chg),
                    "atm_ce_volume": int(ce_vol),
                    "atm_pe_volume": int(pe_vol),
                    "buildup_pattern": buildup["pattern"],
                },
            }
            logger.info(f"Buildup Score: {total_score}/22 | Bias: {bias}")
            return result

        except Exception as e:
            logger.error(f"buildup_score failed: {e}")
            return {
                "score": 0, "max_score": 22, "bias": "UNKNOWN",
                "breakdown": {}, "details": {"error": str(e)},
            }
