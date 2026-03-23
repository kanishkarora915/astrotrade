"""
AstroNifty Engine — Score Engine
Aggregates all module scores, normalizes to 100-scale, classifies bias/confidence.
AstroTrade by Kanishk Arora
"""

from datetime import datetime, timedelta
from loguru import logger

from config import SCORING_WEIGHTS, SIGNAL_THRESHOLDS, INDICES, SECTORS


class ScoreEngine:
    """
    Central scoring engine that consumes raw module scores, normalizes them,
    classifies directional bias, confidence level, and checks cross-index
    consensus across NIFTY / BANKNIFTY / GIFTNIFTY.
    """

    MODULE_MAX = {
        "oi_chain": 25,
        "oi_buildup": 22,
        "astro": 20,
        "greeks": 10,
        "price_action": 10,
        "fii_dii": 8,
        "global_cues": 7,
        "smart_money": 5,
        "expiry": 3,
        "breadth": 2,
    }

    RAW_MAX = sum(MODULE_MAX.values())  # 112

    # ──────────────────────────────────────────────────────────────
    # 1. compute_total_score
    # ──────────────────────────────────────────────────────────────
    def compute_total_score(self, scores: dict, index: str = "NIFTY") -> dict:
        """
        Aggregate raw module scores, normalize to 0-100, classify bias and
        confidence, and return a full result envelope.

        Parameters
        ----------
        scores : dict
            Keys must be module names matching MODULE_MAX.  Values are raw
            numeric scores (clamped to each module's ceiling).
        index : str
            The index this score pertains to (NIFTY / BANKNIFTY / GIFTNIFTY).

        Returns
        -------
        dict with keys: index, total_score, breakdown, bias, confidence,
                        timestamp, valid_till
        """
        try:
            now = datetime.now()

            # Clamp each module score to its allowed maximum and build breakdown
            breakdown = {}
            raw_total = 0.0

            for module, max_val in self.MODULE_MAX.items():
                raw = scores.get(module, 0)
                clamped = max(0.0, min(float(raw), float(max_val)))
                breakdown[module] = {
                    "raw": round(clamped, 2),
                    "max": max_val,
                    "weight_pct": round((max_val / self.RAW_MAX) * 100, 1),
                }
                raw_total += clamped

            # Normalize to 100 scale
            normalized_score = round((raw_total / self.RAW_MAX) * 100, 2)

            bias = self.classify_bias(normalized_score)
            confidence = self.get_confidence(normalized_score)

            result = {
                "index": index,
                "total_score": normalized_score,
                "raw_total": round(raw_total, 2),
                "raw_max": self.RAW_MAX,
                "breakdown": breakdown,
                "bias": bias,
                "confidence": confidence,
                "timestamp": now.isoformat(),
                "valid_till": (now + timedelta(hours=2)).isoformat(),
            }

            logger.info(
                f"[ScoreEngine] {index} — Score {normalized_score}/100 | "
                f"Bias {bias} | Confidence {confidence}"
            )
            return result

        except Exception as e:
            logger.error(f"[ScoreEngine] compute_total_score failed: {e}")
            return {
                "index": index,
                "total_score": 0.0,
                "raw_total": 0.0,
                "raw_max": self.RAW_MAX,
                "breakdown": {},
                "bias": "NEUTRAL",
                "confidence": "LOW",
                "timestamp": datetime.now().isoformat(),
                "valid_till": datetime.now().isoformat(),
                "error": str(e),
            }

    # ──────────────────────────────────────────────────────────────
    # 2. classify_bias
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def classify_bias(score: int) -> str:
        """
        Map a 0-100 normalized score to a directional bias label.

        Thresholds
        ----------
        85+       STRONG_BULL
        70-84     BULL
        55-69     MILD_BULL
        45-54     NEUTRAL
        31-44     MILD_BEAR
        16-30     BEAR
        <16       STRONG_BEAR
        """
        try:
            s = float(score)
            if s >= 85:
                return "STRONG_BULL"
            elif s >= 70:
                return "BULL"
            elif s >= 55:
                return "MILD_BULL"
            elif s >= 45:
                return "NEUTRAL"
            elif s >= 31:
                return "MILD_BEAR"
            elif s >= 16:
                return "BEAR"
            else:
                return "STRONG_BEAR"
        except Exception as e:
            logger.error(f"[ScoreEngine] classify_bias error: {e}")
            return "NEUTRAL"

    # ──────────────────────────────────────────────────────────────
    # 3. get_confidence
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def get_confidence(score: int) -> str:
        """
        Derive confidence level from the normalized score.

        HIGH   — score > 80  (or score < 20, i.e. strong bearish conviction)
        MEDIUM — score 65-80  (or 20-35)
        LOW    — score 55-64  (or 36-44)
        """
        try:
            s = float(score)
            # Use distance from 50 as measure of conviction
            distance = abs(s - 50)
            if distance > 30:
                return "HIGH"
            elif distance >= 15:
                return "MEDIUM"
            else:
                return "LOW"
        except Exception as e:
            logger.error(f"[ScoreEngine] get_confidence error: {e}")
            return "LOW"

    # ──────────────────────────────────────────────────────────────
    # 4. check_cross_index_consensus
    # ──────────────────────────────────────────────────────────────
    def check_cross_index_consensus(self, all_scores: dict) -> dict:
        """
        Check whether NIFTY, BANKNIFTY, and GIFTNIFTY agree on market
        direction.  At least 2 of 3 must share the same broad direction
        for a consensus to form.

        Parameters
        ----------
        all_scores : dict
            Keyed by index name, values are score result dicts (output of
            compute_total_score) containing at minimum a 'bias' key.

        Returns
        -------
        dict with keys: consensus, agreeing_indices, dissenting_indices,
                        strength
        """
        try:
            expected_indices = ["NIFTY", "BANKNIFTY", "GIFTNIFTY"]

            directions = {}
            for idx in expected_indices:
                score_result = all_scores.get(idx)
                if score_result is None:
                    logger.warning(
                        f"[ScoreEngine] consensus check — missing {idx} score"
                    )
                    continue

                bias = score_result.get("bias", "NEUTRAL")

                # Bucket into broad direction
                if bias in ("STRONG_BULL", "BULL", "MILD_BULL"):
                    directions[idx] = "BULL"
                elif bias in ("STRONG_BEAR", "BEAR", "MILD_BEAR"):
                    directions[idx] = "BEAR"
                else:
                    directions[idx] = "NEUTRAL"

            # Count occurrences
            direction_counts = {}
            for idx, direction in directions.items():
                direction_counts.setdefault(direction, []).append(idx)

            # Find majority (need 2 of 3)
            consensus_direction = "MIXED"
            agreeing = []
            dissenting = []

            for direction, indices_list in direction_counts.items():
                if len(indices_list) >= 2 and direction != "NEUTRAL":
                    consensus_direction = direction
                    agreeing = indices_list
                    dissenting = [
                        i for i in directions if i not in indices_list
                    ]
                    break

            # If no clear bull/bear majority, check if 2+ are neutral
            if consensus_direction == "MIXED":
                neutral_list = direction_counts.get("NEUTRAL", [])
                if len(neutral_list) >= 2:
                    consensus_direction = "NEUTRAL"
                    agreeing = neutral_list
                    dissenting = [
                        i for i in directions if i not in neutral_list
                    ]

            # If still mixed, just mark it
            if consensus_direction == "MIXED":
                agreeing = []
                dissenting = list(directions.keys())

            # Strength: STRONG if all 3 agree, MODERATE if 2 agree
            if len(agreeing) == 3:
                strength = "STRONG"
            elif len(agreeing) == 2:
                strength = "MODERATE"
            else:
                strength = "WEAK"

            result = {
                "consensus": consensus_direction,
                "agreeing_indices": agreeing,
                "dissenting_indices": dissenting,
                "strength": strength,
                "individual_directions": directions,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"[ScoreEngine] Consensus: {consensus_direction} "
                f"({strength}) — Agreeing: {agreeing}"
            )
            return result

        except Exception as e:
            logger.error(f"[ScoreEngine] check_cross_index_consensus failed: {e}")
            return {
                "consensus": "MIXED",
                "agreeing_indices": [],
                "dissenting_indices": [],
                "strength": "WEAK",
                "individual_directions": {},
                "error": str(e),
            }

    # ──────────────────────────────────────────────────────────────
    # 5. compute_sector_impact
    # ──────────────────────────────────────────────────────────────
    def compute_sector_impact(self, sector_data: dict) -> dict:
        """
        Analyze which sectors are driving or dragging Nifty, weighted by
        each sector's share in Nifty from config.SECTORS.

        Parameters
        ----------
        sector_data : dict
            Keyed by sector name (matching SECTORS keys).  Each value is a
            dict with at minimum:
                - change_pct : float  (sector's % change today)
            Optionally:
                - oi_bias    : str    ("BULL" / "BEAR" / "NEUTRAL")
                - volume_surge : bool

        Returns
        -------
        dict with keys: impact_score, leading_sectors, lagging_sectors,
                        net_sector_bias, details
        """
        try:
            leading = []
            lagging = []
            details = {}
            weighted_sum = 0.0
            total_weight = 0.0

            for sector_name, sector_conf in SECTORS.items():
                data = sector_data.get(sector_name)
                if data is None:
                    continue

                change_pct = float(data.get("change_pct", 0.0))
                weight = float(sector_conf.get("weight_in_nifty", 0.0))
                oi_bias = data.get("oi_bias", "NEUTRAL")
                volume_surge = data.get("volume_surge", False)

                # Weighted contribution
                weighted_change = change_pct * weight
                weighted_sum += weighted_change
                total_weight += weight

                sector_detail = {
                    "change_pct": round(change_pct, 2),
                    "weight_in_nifty": weight,
                    "weighted_contribution": round(weighted_change, 4),
                    "oi_bias": oi_bias,
                    "volume_surge": volume_surge,
                }
                details[sector_name] = sector_detail

                # Classify leading/lagging using 0.3% threshold
                if change_pct > 0.3:
                    leading.append({
                        "sector": sector_name,
                        "change_pct": round(change_pct, 2),
                        "weighted_contribution": round(weighted_change, 4),
                    })
                elif change_pct < -0.3:
                    lagging.append({
                        "sector": sector_name,
                        "change_pct": round(change_pct, 2),
                        "weighted_contribution": round(weighted_change, 4),
                    })

            # Sort by absolute contribution
            leading.sort(
                key=lambda x: x["weighted_contribution"], reverse=True
            )
            lagging.sort(key=lambda x: x["weighted_contribution"])

            # Net impact score: scale weighted sum to a -10..+10 range
            # Typical weighted sum is roughly -2..+2, so multiply by 5
            impact_score = round(
                max(-10.0, min(10.0, weighted_sum * 5)), 2
            )

            # Net bias
            if impact_score > 2:
                net_bias = "BULL"
            elif impact_score < -2:
                net_bias = "BEAR"
            else:
                net_bias = "NEUTRAL"

            result = {
                "impact_score": impact_score,
                "leading_sectors": leading,
                "lagging_sectors": lagging,
                "net_sector_bias": net_bias,
                "weighted_sum": round(weighted_sum, 4),
                "total_weight_covered": round(total_weight, 2),
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info(
                f"[ScoreEngine] Sector impact {impact_score} | "
                f"Bias {net_bias} | Leading: "
                f"{[s['sector'] for s in leading[:3]]}"
            )
            return result

        except Exception as e:
            logger.error(f"[ScoreEngine] compute_sector_impact failed: {e}")
            return {
                "impact_score": 0.0,
                "leading_sectors": [],
                "lagging_sectors": [],
                "net_sector_bias": "NEUTRAL",
                "details": {},
                "error": str(e),
            }
