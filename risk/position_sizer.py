"""
AstroNifty Position Sizer
-------------------------
Kelly-inspired position sizing with VIX adjustment and margin validation.
"""

import math
from loguru import logger

try:
    from config import RISK_RULES
except ImportError:
    RISK_RULES = {
        "max_lots_first_3_months": 5,
        "max_single_index_pct": 40,
        "vix_reduce_threshold": 18,
        "max_vix": 20,
    }


class PositionSizer:
    """Determine optimal lot count for every trade signal."""

    # ------------------------------------------------------------------
    # 1. Core position sizing  (Kelly-inspired tiers)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_position_size(
        capital: float,
        score: int,
        entry: float,
        sl: float,
        lot_size: int,
    ) -> int:
        """
        Kelly-inspired allocation by score tier.

        Parameters
        ----------
        capital   : available trading capital
        score     : signal confidence score (0-100)
        entry     : planned entry price per unit
        sl        : stop-loss price per unit
        lot_size  : number of units in one lot (e.g. 50 for Nifty)

        Returns
        -------
        int : number of lots to trade (>= 0)
        """
        try:
            if lot_size <= 0 or entry <= 0 or sl <= 0:
                logger.warning(
                    "Invalid inputs: lot_size={}, entry={}, sl={}", lot_size, entry, sl
                )
                return 0

            # --- Allocation tier ---
            if score >= 85:
                allocation_pct = 30.0
            elif score >= 70:
                allocation_pct = 20.0
            elif score >= 60:
                allocation_pct = 10.0
            else:
                logger.info("Score {} below minimum 60; no position", score)
                return 0

            risk_per_trade = capital * (allocation_pct / 100.0)

            # Risk per lot = |entry - sl| * lot_size
            risk_per_lot = abs(entry - sl) * lot_size
            if risk_per_lot <= 0:
                logger.warning("risk_per_lot is zero (entry==sl?); blocking trade")
                return 0

            raw_lots = math.floor(risk_per_trade / risk_per_lot)

            # --- Hard caps ---
            max_lots_cap = RISK_RULES.get("max_lots_first_3_months", 5)
            lots = min(raw_lots, max_lots_cap)

            # 40% single-index cap:  lots * entry * lot_size <= 40% of capital
            max_index_pct = RISK_RULES.get("max_single_index_pct", 40) / 100.0
            max_exposure = capital * max_index_pct
            while lots > 0 and (lots * entry * lot_size) > max_exposure:
                lots -= 1

            lots = max(lots, 0)

            logger.info(
                "position_size | score={} | alloc={}% | risk_per_trade={:.0f} | "
                "risk_per_lot={:.0f} | raw={} | capped={}",
                score,
                allocation_pct,
                risk_per_trade,
                risk_per_lot,
                raw_lots,
                lots,
            )
            return lots

        except Exception as exc:
            logger.exception("calculate_position_size crashed: {}", exc)
            return 0

    # ------------------------------------------------------------------
    # 2. VIX-based adjustment
    # ------------------------------------------------------------------
    @staticmethod
    def adjust_for_vix(lots: int, vix: float) -> int:
        """
        Reduce or block lots based on India VIX level.

        VIX 0-15  : full lots (unchanged)
        VIX 15-18 : full lots
        VIX 18-20 : reduce by 50%
        VIX > 20  : 0 lots (block)

        Returns
        -------
        int : adjusted lot count
        """
        try:
            vix_reduce = RISK_RULES.get("vix_reduce_threshold", 18)
            vix_max = RISK_RULES.get("max_vix", 20)

            if vix > vix_max:
                logger.warning("VIX {:.2f} > {} | BLOCKING trade (0 lots)", vix, vix_max)
                return 0

            if vix >= vix_reduce:
                adjusted = max(1, math.floor(lots * 0.5))
                logger.info(
                    "VIX {:.2f} in caution zone ({}-{}) | lots {} -> {}",
                    vix,
                    vix_reduce,
                    vix_max,
                    lots,
                    adjusted,
                )
                return adjusted

            # VIX < 18: no change
            logger.debug("VIX {:.2f} normal | lots unchanged at {}", vix, lots)
            return lots

        except Exception as exc:
            logger.exception("adjust_for_vix crashed: {}", exc)
            return 0

    # ------------------------------------------------------------------
    # 3. Margin validation
    # ------------------------------------------------------------------
    @staticmethod
    def validate_margin(
        lots: int,
        entry: float,
        lot_size: int,
        available_margin: float,
    ) -> int:
        """
        Ensure total exposure does not exceed available margin.
        Reduces lots until the position fits.

        Parameters
        ----------
        lots             : desired lot count
        entry            : entry price per unit
        lot_size         : units per lot
        available_margin : broker-reported available margin

        Returns
        -------
        int : number of lots that fit within margin
        """
        try:
            if lots <= 0 or entry <= 0 or lot_size <= 0:
                return 0

            exposure_per_lot = entry * lot_size
            if exposure_per_lot <= 0:
                return 0

            max_affordable = math.floor(available_margin / exposure_per_lot)
            validated = min(lots, max_affordable)
            validated = max(validated, 0)

            if validated < lots:
                logger.warning(
                    "Margin constraint | requested={} lots | affordable={} lots | "
                    "margin={:.0f} | exposure_per_lot={:.0f}",
                    lots,
                    validated,
                    available_margin,
                    exposure_per_lot,
                )
            else:
                logger.debug(
                    "Margin OK | lots={} | margin={:.0f} | exposure={:.0f}",
                    validated,
                    available_margin,
                    validated * exposure_per_lot,
                )

            return validated

        except Exception as exc:
            logger.exception("validate_margin crashed: {}", exc)
            return 0
