"""
AstroNifty Pattern Matcher
Finds historically similar astro + OI patterns and computes win-rate
probabilities for call / put bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from loguru import logger


class PatternMatcher:
    """Match current astro / OI snapshot against historical patterns."""

    # Weights for the combined probability model
    ASTRO_WEIGHT = 0.55
    OI_WEIGHT = 0.45

    def __init__(self, db_manager):
        """
        Parameters
        ----------
        db_manager : object
            Must expose ``execute_query(sql, params)`` returning list[dict]
            and ``get_session()`` context-manager for ORM queries.
        """
        self.db = db_manager
        logger.info("PatternMatcher initialised")

    # ------------------------------------------------------------------
    # 1. Astro pattern matching
    # ------------------------------------------------------------------

    def find_similar_patterns(
        self,
        current_astro: dict,
        lookback_days: int = 365,
    ) -> list[dict]:
        """Find historical dates whose nakshatra + tithi + hora match the current day.

        Parameters
        ----------
        current_astro : dict
            Required keys: ``nakshatra``, ``tithi``, ``hora``.
            Optional: ``yoga``, ``karana``.
        lookback_days : int
            How far back to search.

        Returns
        -------
        list[dict]  each item: date, nakshatra, tithi, hora, outcome (bull/bear/neutral),
                    close_change_pct.
        """
        nakshatra = current_astro.get("nakshatra", "")
        tithi = current_astro.get("tithi", "")
        hora = current_astro.get("hora", "")
        cutoff = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        query = """
            SELECT ad.date, ad.nakshatra, ad.tithi, ad.hora,
                   md.open, md.close
            FROM astro_data ad
            JOIN market_data md ON ad.date = md.date
            WHERE ad.date >= :cutoff
        """
        params = {"cutoff": cutoff}

        rows = self.db.execute_query(query, params)
        if not rows:
            logger.warning("No historical astro+market data found in last {} days", lookback_days)
            return []

        matches: list[dict] = []
        for row in rows:
            score = 0
            if row.get("nakshatra") == nakshatra:
                score += 3
            if row.get("tithi") == tithi:
                score += 2
            if row.get("hora") == hora:
                score += 1

            if score == 0:
                continue

            open_price = float(row.get("open", 0))
            close_price = float(row.get("close", 0))
            change_pct = ((close_price - open_price) / open_price * 100.0) if open_price else 0.0

            if change_pct > 0.25:
                outcome = "bull"
            elif change_pct < -0.25:
                outcome = "bear"
            else:
                outcome = "neutral"

            matches.append({
                "date": row["date"],
                "nakshatra": row.get("nakshatra"),
                "tithi": row.get("tithi"),
                "hora": row.get("hora"),
                "match_score": score,
                "close_change_pct": round(change_pct, 4),
                "outcome": outcome,
            })

        matches.sort(key=lambda m: m["match_score"], reverse=True)
        logger.info(
            "Found {} similar astro patterns (nakshatra={}, tithi={}, hora={})",
            len(matches), nakshatra, tithi, hora,
        )
        return matches

    # ------------------------------------------------------------------
    # 2. Pattern win-rate from backtest table
    # ------------------------------------------------------------------

    def calculate_pattern_win_rate(self, pattern_name: str) -> dict:
        """Query the PatternBacktest table for aggregate stats on *pattern_name*.

        Returns
        -------
        dict: total_occurrences, bull_rate, bear_rate, avg_return
        """
        query = """
            SELECT outcome, return_pct
            FROM pattern_backtest
            WHERE pattern_name = :pattern_name
        """
        rows = self.db.execute_query(query, {"pattern_name": pattern_name})

        if not rows:
            logger.warning("No backtest records for pattern '{}'", pattern_name)
            return {
                "total_occurrences": 0,
                "bull_rate": 0.0,
                "bear_rate": 0.0,
                "avg_return": 0.0,
            }

        outcomes = [r["outcome"] for r in rows]
        returns = [float(r.get("return_pct", 0)) for r in rows]
        counts = Counter(outcomes)
        total = len(outcomes)

        result = {
            "total_occurrences": total,
            "bull_rate": round(counts.get("bull", 0) / total, 4),
            "bear_rate": round(counts.get("bear", 0) / total, 4),
            "avg_return": round(float(np.mean(returns)), 4) if returns else 0.0,
        }
        logger.info("Pattern '{}' win-rate: {}", pattern_name, result)
        return result

    # ------------------------------------------------------------------
    # 3. OI distribution matching
    # ------------------------------------------------------------------

    def match_oi_pattern(
        self,
        current_chain: pd.DataFrame,
        historical_chains: list[pd.DataFrame],
    ) -> dict:
        """Find the most similar historical OI distribution via cosine similarity.

        Parameters
        ----------
        current_chain : pd.DataFrame
            Columns: strike, call_oi, put_oi.
        historical_chains : list[pd.DataFrame]
            Each element has the same schema plus a ``date`` and ``outcome`` column.

        Returns
        -------
        dict: similarity_score (0-1), historical_outcome, matched_date.
        """
        if current_chain.empty or not historical_chains:
            logger.warning("Insufficient data for OI pattern matching")
            return {"similarity_score": 0.0, "historical_outcome": "neutral", "matched_date": None}

        current_vec = self._oi_to_vector(current_chain)
        if current_vec is None:
            return {"similarity_score": 0.0, "historical_outcome": "neutral", "matched_date": None}

        best_sim = -1.0
        best_outcome = "neutral"
        best_date = None

        for hist_df in historical_chains:
            hist_vec = self._oi_to_vector(hist_df)
            if hist_vec is None or len(hist_vec) != len(current_vec):
                continue
            sim = self._cosine_similarity(current_vec, hist_vec)
            if sim > best_sim:
                best_sim = sim
                best_outcome = hist_df["outcome"].iloc[0] if "outcome" in hist_df.columns else "neutral"
                best_date = hist_df["date"].iloc[0] if "date" in hist_df.columns else None

        result = {
            "similarity_score": round(max(best_sim, 0.0), 4),
            "historical_outcome": best_outcome,
            "matched_date": best_date,
        }
        logger.info("Best OI pattern match: sim={:.4f}, outcome={}", result["similarity_score"], best_outcome)
        return result

    # ------------------------------------------------------------------
    # 4. Combined probability
    # ------------------------------------------------------------------

    def get_combined_probability(self, astro_match: dict, oi_match: dict) -> dict:
        """Weighted combination of astro and OI pattern signals.

        Parameters
        ----------
        astro_match : dict
            Output of ``find_similar_patterns`` aggregated, or at minimum
            keys: ``bull_rate``, ``bear_rate``.
        oi_match : dict
            Output of ``match_oi_pattern`` -- keys: ``similarity_score``,
            ``historical_outcome``.

        Returns
        -------
        dict: call_prob, put_prob, confidence.
        """
        # --- Astro component ---
        astro_bull = float(astro_match.get("bull_rate", 0.5))
        astro_bear = float(astro_match.get("bear_rate", 0.5))

        # --- OI component ---
        oi_sim = float(oi_match.get("similarity_score", 0.0))
        oi_outcome = oi_match.get("historical_outcome", "neutral")
        if oi_outcome == "bull":
            oi_bull, oi_bear = 0.7, 0.3
        elif oi_outcome == "bear":
            oi_bull, oi_bear = 0.3, 0.7
        else:
            oi_bull, oi_bear = 0.5, 0.5

        # Scale OI signal by similarity strength
        oi_bull = 0.5 + (oi_bull - 0.5) * oi_sim
        oi_bear = 0.5 + (oi_bear - 0.5) * oi_sim

        # --- Weighted blend ---
        w_a = self.ASTRO_WEIGHT
        w_o = self.OI_WEIGHT
        call_prob = w_a * astro_bull + w_o * oi_bull
        put_prob = w_a * astro_bear + w_o * oi_bear

        # Normalise so they sum to 1
        total = call_prob + put_prob
        if total > 0:
            call_prob /= total
            put_prob /= total

        confidence = abs(call_prob - put_prob)

        result = {
            "call_prob": round(call_prob, 4),
            "put_prob": round(put_prob, 4),
            "confidence": round(confidence, 4),
        }
        logger.info("Combined probability: call={:.2%} put={:.2%} conf={:.2%}", call_prob, put_prob, confidence)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _oi_to_vector(chain_df: pd.DataFrame) -> np.ndarray | None:
        """Convert OI chain to a normalised float vector [call_oi..., put_oi...]."""
        try:
            calls = chain_df["call_oi"].astype(float).values
            puts = chain_df["put_oi"].astype(float).values
            vec = np.concatenate([calls, puts])
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else None
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / denom if denom > 0 else 0.0
