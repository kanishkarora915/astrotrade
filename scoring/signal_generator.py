"""
AstroNifty Engine — Signal Generator
Produces actionable CALL_BUY / PUT_BUY / CE_SELL / PE_SELL signals
when the composite score crosses the entry threshold.
AstroTrade by Kanishk Arora
"""

import uuid
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger

from config import INDICES, RISK_RULES
from scoring.strike_selector import StrikeSelector


class SignalGenerator:
    """
    Converts a ScoreEngine result into a fully-specified trading signal
    with strike, SL, target, risk-reward, and Kite-compatible instrument
    symbol.
    """

    ENTRY_THRESHOLD = 70  # Minimum normalized score to generate a signal

    def __init__(self):
        self.strike_selector = StrikeSelector()

    # ──────────────────────────────────────────────────────────────
    # 1. generate_signal
    # ──────────────────────────────────────────────────────────────
    def generate_signal(
        self,
        score_result: dict,
        chain: pd.DataFrame,
        spot: float,
        index: str,
    ) -> dict | None:
        """
        Produce a trade signal if the score meets the entry threshold.

        Parameters
        ----------
        score_result : dict
            Output of ScoreEngine.compute_total_score().
        chain : pd.DataFrame
            Live option chain with columns: strike, CE_LTP, PE_LTP,
            CE_delta, PE_delta, expiry (at minimum).
        spot : float
            Current spot price of the index.
        index : str
            Index name (NIFTY / BANKNIFTY / GIFTNIFTY).

        Returns
        -------
        dict  — full signal envelope, or None if score is below threshold.
        """
        try:
            total_score = float(score_result.get("total_score", 0))
            bias = score_result.get("bias", "NEUTRAL")
            confidence = score_result.get("confidence", "LOW")

            if total_score < self.ENTRY_THRESHOLD:
                logger.info(
                    f"[SignalGenerator] {index} score {total_score} below "
                    f"threshold {self.ENTRY_THRESHOLD} — no signal"
                )
                return None

            # Determine signal type
            signal_type = self.determine_signal_type(bias, total_score)
            option_type = self._option_type_from_signal(signal_type)

            # Select strike
            strike = self.strike_selector.select_strike(
                spot, index, signal_type, int(total_score)
            )

            # Get entry price from chain
            entry_price = self.strike_selector.get_entry_price(
                chain, strike, option_type
            )
            if entry_price <= 0:
                logger.warning(
                    f"[SignalGenerator] Could not find valid entry price "
                    f"for {index} {strike}{option_type}"
                )
                return None

            # Get delta for SL/target calculation
            delta = self._get_delta_from_chain(chain, strike, option_type)

            # Calculate SL and target
            sl_price, target_price, rr_ratio = (
                self.strike_selector.calculate_sl_target(
                    entry_price, delta, signal_type
                )
            )

            # Determine expiry from chain
            expiry = self._extract_nearest_expiry(chain)

            # Format the Kite instrument symbol
            instrument_symbol = self.format_instrument_symbol(
                index, expiry, strike, option_type
            )

            # Lot size
            idx_config = INDICES.get(index, INDICES["NIFTY"])
            lot_size = idx_config["lot_size"]

            # Astro window from score breakdown (if astro module contributed)
            astro_window = self._extract_astro_window(score_result)

            # Build reason list from top contributing modules
            reasons = self._build_reasons(score_result, bias)

            # Build avoid_if list (conditions under which to skip)
            avoid_if = self._build_avoid_conditions(
                score_result, index, spot
            )

            now = datetime.now()
            validity_mins = RISK_RULES.get("signal_validity_mins", 120)

            signal = {
                "signal_id": str(uuid.uuid4()),
                "timestamp": now.isoformat(),
                "index": index,
                "signal_type": signal_type,
                "expiry": expiry,
                "strike": strike,
                "option_type": option_type,
                "instrument_symbol": instrument_symbol,
                "entry_price": round(entry_price, 2),
                "entry_type": "MARKET" if confidence == "HIGH" else "LIMIT",
                "stop_loss": round(sl_price, 2),
                "target": round(target_price, 2),
                "risk_reward": round(rr_ratio, 2),
                "quantity": lot_size,
                "score": round(total_score, 2),
                "bias": bias,
                "confidence": confidence,
                "astro_window": astro_window,
                "reasons": reasons,
                "avoid_if": avoid_if,
                "valid_till": (
                    now + timedelta(minutes=validity_mins)
                ).isoformat(),
            }

            logger.info(
                f"[SignalGenerator] SIGNAL — {signal_type} {instrument_symbol} "
                f"@ {entry_price} | SL {sl_price} | TGT {target_price} | "
                f"RR {rr_ratio}"
            )
            return signal

        except Exception as e:
            logger.error(f"[SignalGenerator] generate_signal failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # 2. determine_signal_type
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def determine_signal_type(bias: str, score: int) -> str:
        """
        Map bias + confidence into a signal type.

        Directional buys for strong conviction, OTM sells for very high
        confidence on the opposite side.

        Returns
        -------
        str — one of CALL_BUY, PUT_BUY, CE_SELL, PE_SELL
        """
        try:
            s = float(score)

            # Strong bullish conviction with very high score -> sell OTM puts
            if bias in ("STRONG_BULL",) and s >= 90:
                return "PE_SELL"

            # Bullish
            if bias in ("STRONG_BULL", "BULL", "MILD_BULL"):
                return "CALL_BUY"

            # Strong bearish conviction with very high inverse score -> sell OTM calls
            if bias in ("STRONG_BEAR",) and s <= 10:
                return "CE_SELL"

            # Bearish
            if bias in ("STRONG_BEAR", "BEAR", "MILD_BEAR"):
                return "PUT_BUY"

            # Neutral — no signal (shouldn't reach here if threshold enforced)
            return "CALL_BUY"

        except Exception as e:
            logger.error(f"[SignalGenerator] determine_signal_type error: {e}")
            return "CALL_BUY"

    # ──────────────────────────────────────────────────────────────
    # 3. format_instrument_symbol
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def format_instrument_symbol(
        index: str, expiry: str, strike: int, option_type: str
    ) -> str:
        """
        Generate Kite-compatible trading symbol.

        Format: NIFTY24APR23900CE
                {INDEX}{YY}{MMM}{STRIKE}{CE/PE}

        Parameters
        ----------
        index : str       — "NIFTY" / "BANKNIFTY"
        expiry : str      — ISO date string "2024-04-25" or "25APR2024"
        strike : int      — Strike price
        option_type : str — "CE" or "PE"

        Returns
        -------
        str — Kite instrument symbol
        """
        try:
            # Parse expiry into components
            if "-" in expiry:
                dt = datetime.strptime(expiry, "%Y-%m-%d")
            else:
                # Try common formats
                for fmt in ("%d%b%Y", "%d-%b-%Y", "%d/%m/%Y"):
                    try:
                        dt = datetime.strptime(expiry.upper(), fmt)
                        break
                    except ValueError:
                        continue
                else:
                    dt = datetime.strptime(expiry[:10], "%Y-%m-%d")

            yy = dt.strftime("%y")       # "24"
            mmm = dt.strftime("%b").upper()  # "APR"
            day = dt.strftime("%d")       # For weekly expiries

            # Map index name to Kite symbol prefix
            prefix_map = {
                "NIFTY": "NIFTY",
                "BANKNIFTY": "BANKNIFTY",
                "GIFTNIFTY": "NIFTY",  # Gift Nifty options trade under NIFTY
            }
            prefix = prefix_map.get(index, index)

            # Weekly: NIFTY2441723900CE  (YY + M_code + DD)
            # Monthly: NIFTY24APR23900CE
            # Use monthly format for simplicity (matches most brokers)
            symbol = f"{prefix}{yy}{mmm}{int(strike)}{option_type.upper()}"

            return symbol

        except Exception as e:
            logger.error(
                f"[SignalGenerator] format_instrument_symbol error: {e}"
            )
            return f"{index}{int(strike)}{option_type.upper()}"

    # ══════════════════════════════════════════════════════════════
    # PRIVATE HELPERS
    # ══════════════════════════════════════════════════════════════

    @staticmethod
    def _option_type_from_signal(signal_type: str) -> str:
        """Map signal type to option type CE/PE."""
        if signal_type in ("CALL_BUY", "CE_SELL"):
            return "CE"
        return "PE"

    @staticmethod
    def _get_delta_from_chain(
        chain: pd.DataFrame, strike: int, option_type: str
    ) -> float:
        """Extract delta for a given strike from the option chain."""
        try:
            row = chain.loc[chain["strike"] == strike]
            if row.empty:
                return 0.5  # default mid-delta

            delta_col = f"{option_type}_delta"
            if delta_col in row.columns:
                return float(row.iloc[0][delta_col])

            # Fallback: estimate from moneyness
            return 0.5

        except Exception as e:
            logger.warning(
                f"[SignalGenerator] _get_delta_from_chain fallback: {e}"
            )
            return 0.5

    @staticmethod
    def _extract_nearest_expiry(chain: pd.DataFrame) -> str:
        """Get the nearest expiry date string from the chain."""
        try:
            if "expiry" not in chain.columns:
                # Return next Thursday as default
                today = datetime.now()
                days_ahead = (3 - today.weekday()) % 7  # Thursday = 3
                if days_ahead == 0 and today.hour >= 15:
                    days_ahead = 7
                nearest = today + timedelta(days=days_ahead)
                return nearest.strftime("%Y-%m-%d")

            expiries = pd.to_datetime(chain["expiry"]).sort_values().unique()
            nearest = pd.Timestamp(expiries[0])
            return nearest.strftime("%Y-%m-%d")

        except Exception as e:
            logger.warning(
                f"[SignalGenerator] _extract_nearest_expiry fallback: {e}"
            )
            today = datetime.now()
            days_ahead = (3 - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    @staticmethod
    def _extract_astro_window(score_result: dict) -> dict:
        """Pull astro timing window from the score breakdown if available."""
        try:
            breakdown = score_result.get("breakdown", {})
            astro_info = breakdown.get("astro", {})
            return {
                "active": astro_info.get("raw", 0) > 10,
                "astro_score": astro_info.get("raw", 0),
                "astro_max": astro_info.get("max", 20),
            }
        except Exception:
            return {"active": False, "astro_score": 0, "astro_max": 20}

    @staticmethod
    def _build_reasons(score_result: dict, bias: str) -> list:
        """
        Build a human-readable list of reasons supporting this signal
        by identifying the top contributing modules.
        """
        try:
            reasons = []
            breakdown = score_result.get("breakdown", {})

            # Sort modules by their raw contribution percentage
            scored_modules = []
            for module, info in breakdown.items():
                raw = info.get("raw", 0)
                max_val = info.get("max", 1)
                pct_of_max = (raw / max_val * 100) if max_val > 0 else 0
                scored_modules.append((module, raw, max_val, pct_of_max))

            scored_modules.sort(key=lambda x: x[3], reverse=True)

            module_labels = {
                "oi_chain": "OI chain analysis",
                "oi_buildup": "OI buildup pattern",
                "astro": "Astro timing window",
                "greeks": "Greeks alignment",
                "price_action": "Price action structure",
                "fii_dii": "FII/DII flow",
                "global_cues": "Global market cues",
                "smart_money": "Smart money positioning",
                "expiry": "Expiry dynamics",
                "breadth": "Market breadth",
            }

            direction = "bullish" if "BULL" in bias else (
                "bearish" if "BEAR" in bias else "neutral"
            )

            for module, raw, max_val, pct in scored_modules[:5]:
                if pct >= 60:
                    label = module_labels.get(module, module)
                    reasons.append(
                        f"{label} strongly {direction} ({raw}/{max_val})"
                    )
                elif pct >= 40:
                    label = module_labels.get(module, module)
                    reasons.append(
                        f"{label} mildly {direction} ({raw}/{max_val})"
                    )

            if not reasons:
                reasons.append(
                    f"Composite score {score_result.get('total_score', 0)} "
                    f"above entry threshold"
                )

            return reasons

        except Exception as e:
            logger.warning(f"[SignalGenerator] _build_reasons fallback: {e}")
            return ["Composite score above entry threshold"]

    @staticmethod
    def _build_avoid_conditions(
        score_result: dict, index: str, spot: float
    ) -> list:
        """
        Build a list of conditions under which the signal should NOT be
        taken, based on weak modules or risk rules.
        """
        try:
            avoid = []
            breakdown = score_result.get("breakdown", {})
            confidence = score_result.get("confidence", "LOW")

            # Weak OI chain
            oi_info = breakdown.get("oi_chain", {})
            if oi_info.get("raw", 0) < 8:
                avoid.append("OI chain data is weak — wait for confirmation")

            # Weak price action
            pa_info = breakdown.get("price_action", {})
            if pa_info.get("raw", 0) < 3:
                avoid.append(
                    "Price action not confirming — avoid near resistance/support"
                )

            # Low confidence
            if confidence == "LOW":
                avoid.append("Confidence is LOW — reduce position size by 50%")

            # Late entry warning
            no_entry_after = RISK_RULES.get("no_entry_after", "15:00")
            avoid.append(
                f"Do not enter after {no_entry_after} IST"
            )

            # VIX warning
            vix_reduce = RISK_RULES.get("vix_reduce_above", 18)
            vix_block = RISK_RULES.get("vix_block_above", 20)
            avoid.append(
                f"Reduce size if VIX > {vix_reduce}, skip if VIX > {vix_block}"
            )

            return avoid

        except Exception as e:
            logger.warning(
                f"[SignalGenerator] _build_avoid_conditions fallback: {e}"
            )
            return ["Verify market conditions before entry"]
