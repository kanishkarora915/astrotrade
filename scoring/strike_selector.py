"""
AstroNifty Engine — Strike Selector
Picks optimal strike, calculates SL/target with dual-method approach,
and reads entry prices from the live option chain.
AstroTrade by Kanishk Arora
"""

import math

import pandas as pd
from loguru import logger

from config import INDICES


class StrikeSelector:
    """
    Handles strike selection based on score confidence, SL/target
    calculation using both percentage and delta methods, and entry
    price extraction from the option chain DataFrame.
    """

    # ──────────────────────────────────────────────────────────────
    # 1. select_strike
    # ──────────────────────────────────────────────────────────────
    def select_strike(
        self, spot: float, index: str, signal_type: str, score: int
    ) -> int:
        """
        Select the optimal strike based on score confidence level and
        signal direction.

        Strike selection logic
        ----------------------
        Score 85+  → ATM strike (nearest to spot)
        Score 70-84 → 1 strike OTM
        Score 65-69 → 2 strikes OTM

        For CE signals: round spot UP to nearest strike_gap
        For PE signals: round spot DOWN to nearest strike_gap

        Parameters
        ----------
        spot : float       — Current spot price
        index : str        — Index name (NIFTY / BANKNIFTY / GIFTNIFTY)
        signal_type : str  — CALL_BUY / PUT_BUY / CE_SELL / PE_SELL
        score : int        — Normalized score (0-100)

        Returns
        -------
        int — Selected strike price
        """
        try:
            idx_config = INDICES.get(index, INDICES["NIFTY"])
            strike_gap = idx_config["strike_gap"]

            # Determine direction from signal type
            is_ce = signal_type in ("CALL_BUY", "CE_SELL")

            # Calculate ATM strike
            if is_ce:
                # For CE: round up to nearest strike gap
                atm_strike = int(math.ceil(spot / strike_gap) * strike_gap)
            else:
                # For PE: round down to nearest strike gap
                atm_strike = int(math.floor(spot / strike_gap) * strike_gap)

            # Determine OTM offset based on score
            if score >= 85:
                otm_steps = 0  # ATM
            elif score >= 70:
                otm_steps = 1  # 1 strike OTM
            else:
                otm_steps = 2  # 2 strikes OTM (score 65-69)

            # Apply OTM offset in the correct direction
            if is_ce:
                # OTM for CE means higher strike
                selected_strike = atm_strike + (otm_steps * strike_gap)
            else:
                # OTM for PE means lower strike
                selected_strike = atm_strike - (otm_steps * strike_gap)

            logger.info(
                f"[StrikeSelector] {index} spot={spot} | "
                f"ATM={atm_strike} | OTM_steps={otm_steps} | "
                f"Selected={selected_strike} ({signal_type})"
            )
            return selected_strike

        except Exception as e:
            logger.error(f"[StrikeSelector] select_strike failed: {e}")
            # Fallback: round to nearest strike gap
            idx_config = INDICES.get(index, INDICES["NIFTY"])
            strike_gap = idx_config.get("strike_gap", 50)
            return int(round(spot / strike_gap) * strike_gap)

    # ──────────────────────────────────────────────────────────────
    # 2. calculate_sl_target
    # ──────────────────────────────────────────────────────────────
    def calculate_sl_target(
        self, entry: float, delta: float, signal_type: str
    ) -> tuple:
        """
        Compute stop-loss and target using dual-method approach, then
        pick the more conservative value for each.

        Stop-Loss (use the HIGHER / tighter SL)
        -----------------------------------------
        Method 1: entry * 0.75  — lose maximum 25% of premium
        Method 2: entry - (|delta| * 50) — 50-point adverse index move

        Target (use the LOWER / more conservative target)
        --------------------------------------------------
        Method 1: entry * 1.50  — gain 50% on premium
        Method 2: entry + (|delta| * 100) — 100-point favorable index move

        For SELL signals the logic is inverted (SL is above entry, target
        below entry).

        Parameters
        ----------
        entry : float       — Entry price (premium)
        delta : float       — Option delta (absolute or signed)
        signal_type : str   — CALL_BUY / PUT_BUY / CE_SELL / PE_SELL

        Returns
        -------
        tuple of (sl_price, target_price, risk_reward_ratio)
        """
        try:
            abs_delta = abs(delta) if delta != 0 else 0.5
            is_sell = signal_type in ("CE_SELL", "PE_SELL")

            if not is_sell:
                # ── BUY SIGNALS ──────────────────────────────────
                # Stop-loss: pick the HIGHER value (tighter stop)
                sl_method_1 = entry * 0.75
                sl_method_2 = entry - (abs_delta * 50)
                sl_price = max(sl_method_1, sl_method_2)
                # Floor SL at 1 (premium can't go below 0.05 realistically)
                sl_price = max(sl_price, 1.0)

                # Target: pick the LOWER value (conservative target)
                tgt_method_1 = entry * 1.50
                tgt_method_2 = entry + (abs_delta * 100)
                target_price = min(tgt_method_1, tgt_method_2)

            else:
                # ── SELL SIGNALS (short premium) ─────────────────
                # For sells, SL is ABOVE entry (premium rises against us)
                sl_method_1 = entry * 1.25   # premium rises 25%
                sl_method_2 = entry + (abs_delta * 50)
                sl_price = min(sl_method_1, sl_method_2)  # tighter = lower

                # Target for sells is BELOW entry (premium decays)
                tgt_method_1 = entry * 0.50   # premium falls 50%
                tgt_method_2 = entry - (abs_delta * 100)
                target_price = max(tgt_method_1, tgt_method_2)  # conservative
                target_price = max(target_price, 0.05)  # floor

            # Risk-reward ratio
            risk = abs(entry - sl_price)
            reward = abs(target_price - entry)
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

            logger.info(
                f"[StrikeSelector] SL/TGT — Entry={entry} | "
                f"SL={round(sl_price, 2)} | TGT={round(target_price, 2)} | "
                f"RR={rr_ratio} | Delta={abs_delta} | Type={signal_type}"
            )
            return (round(sl_price, 2), round(target_price, 2), rr_ratio)

        except Exception as e:
            logger.error(
                f"[StrikeSelector] calculate_sl_target failed: {e}"
            )
            # Emergency fallback: 25% SL, 50% target
            sl_fallback = round(entry * 0.75, 2)
            tgt_fallback = round(entry * 1.50, 2)
            risk = entry - sl_fallback
            reward = tgt_fallback - entry
            rr_fallback = round(reward / risk, 2) if risk > 0 else 2.0
            return (sl_fallback, tgt_fallback, rr_fallback)

    # ──────────────────────────────────────────────────────────────
    # 3. get_entry_price
    # ──────────────────────────────────────────────────────────────
    def get_entry_price(
        self, chain: pd.DataFrame, strike: int, option_type: str
    ) -> float:
        """
        Look up the last traded price (LTP) from the option chain for
        the given strike and option type.

        Parameters
        ----------
        chain : pd.DataFrame
            Must contain columns: strike, CE_LTP, PE_LTP
        strike : int
            The strike price to look up.
        option_type : str
            "CE" or "PE"

        Returns
        -------
        float — LTP for the selected strike/type, or 0.0 if not found.
        """
        try:
            if chain is None or chain.empty:
                logger.warning(
                    "[StrikeSelector] get_entry_price — chain is empty"
                )
                return 0.0

            ltp_col = f"{option_type}_LTP"

            # Ensure strike column exists
            if "strike" not in chain.columns:
                logger.warning(
                    "[StrikeSelector] get_entry_price — 'strike' column "
                    "not found in chain"
                )
                return 0.0

            if ltp_col not in chain.columns:
                # Try alternative column names
                alt_cols = {
                    "CE_LTP": ["CE_ltp", "ce_ltp", "call_ltp", "CE_last_price"],
                    "PE_LTP": ["PE_ltp", "pe_ltp", "put_ltp", "PE_last_price"],
                }
                found = False
                for alt in alt_cols.get(ltp_col, []):
                    if alt in chain.columns:
                        ltp_col = alt
                        found = True
                        break
                if not found:
                    logger.warning(
                        f"[StrikeSelector] get_entry_price — column "
                        f"'{ltp_col}' not found in chain. "
                        f"Available: {list(chain.columns)}"
                    )
                    return 0.0

            # Filter for the strike
            row = chain.loc[chain["strike"] == strike]
            if row.empty:
                # Try nearest strike if exact match not found
                chain_strikes = chain["strike"].values
                if len(chain_strikes) == 0:
                    return 0.0
                nearest_idx = (abs(chain_strikes - strike)).argmin()
                row = chain.iloc[[nearest_idx]]
                logger.info(
                    f"[StrikeSelector] Exact strike {strike} not found, "
                    f"using nearest {int(chain_strikes[nearest_idx])}"
                )

            ltp = float(row.iloc[0][ltp_col])

            if ltp <= 0:
                logger.warning(
                    f"[StrikeSelector] LTP for {strike}{option_type} is "
                    f"{ltp} — likely stale data"
                )
                return 0.0

            logger.debug(
                f"[StrikeSelector] {strike}{option_type} LTP = {ltp}"
            )
            return round(ltp, 2)

        except Exception as e:
            logger.error(f"[StrikeSelector] get_entry_price failed: {e}")
            return 0.0
