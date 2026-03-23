"""
AstroNifty Pre-Trade Checklist
-------------------------------
10-point validation that every signal must clear before reaching the
execution layer.  Independent of RiskManager so it can also be used
for paper-trade scoring and back-test filtering.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytz
from loguru import logger

IST = pytz.timezone("Asia/Kolkata")

# Thresholds (can be overridden via config injection)
DEFAULTS = {
    "min_score": 70,
    "min_rr_ratio": 1.5,
    "max_iv_rank": 80,
    "event_lookahead_hours": 2,
    "max_vix": 20,
    "vix_warn": 18,
    "max_spread_pct": 5.0,
    "min_volume": 1000,
    "min_hours_to_expiry": 2,
    "max_capital_alloc_pct": 40.0,
}


class PreTradeChecklist:
    """Run a 10-point pre-trade validation and return a structured report."""

    def __init__(self, overrides: Dict[str, Any] | None = None) -> None:
        self.cfg = {**DEFAULTS, **(overrides or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_checklist(self, signal: dict, market_state: dict) -> dict:
        """
        Validate *signal* against current *market_state*.

        Parameters
        ----------
        signal : dict
            Required keys:
                score (int), entry (float), target (float), sl (float),
                iv_rank (float), premium (float), bid (float), ask (float),
                volume (int), expiry (datetime | str ISO),
                capital_allocation_pct (float)
        market_state : dict
            Required keys:
                vix (float),
                upcoming_events (list[dict] with 'time' key as datetime),
                circuit_breaker_active (bool)

        Returns
        -------
        dict:
            passed (bool)        - True only if ALL 10 checks pass
            score  (int)         - number of checks passed (0-10)
            failed_checks (list) - names of failed checks
            warnings (list)      - non-blocking advisory messages
        """
        try:
            passed_checks: List[str] = []
            failed_checks: List[str] = []
            warnings: List[str] = []

            # --- 1. Score threshold ---
            self._check_score(signal, passed_checks, failed_checks)

            # --- 2. Risk-reward ratio >= 1.5 ---
            self._check_risk_reward(signal, passed_checks, failed_checks, warnings)

            # --- 3. IV not extreme (rank < 80 for buying) ---
            self._check_iv_rank(signal, passed_checks, failed_checks, warnings)

            # --- 4. No earnings/event in next 2 hours ---
            self._check_upcoming_events(market_state, passed_checks, failed_checks, warnings)

            # --- 5. VIX acceptable ---
            self._check_vix(market_state, passed_checks, failed_checks, warnings)

            # --- 6. Spread (bid-ask) < 5% of premium ---
            self._check_spread(signal, passed_checks, failed_checks, warnings)

            # --- 7. Volume adequate (>1000 contracts at strike) ---
            self._check_volume(signal, passed_checks, failed_checks, warnings)

            # --- 8. Time not too close to expiry (>2 hours for buying) ---
            self._check_time_to_expiry(signal, passed_checks, failed_checks, warnings)

            # --- 9. No circuit breaker risk ---
            self._check_circuit_breaker(market_state, passed_checks, failed_checks)

            # --- 10. Capital allocation within limits ---
            self._check_capital_allocation(signal, passed_checks, failed_checks)

            # --- Aggregate ---
            total_passed = len(passed_checks)
            all_passed = len(failed_checks) == 0

            result = {
                "passed": all_passed,
                "score": total_passed,
                "failed_checks": failed_checks,
                "warnings": warnings,
            }

            log_fn = logger.info if all_passed else logger.warning
            log_fn(
                "PreTradeChecklist | passed={} | score={}/10 | failed={} | warnings={}",
                all_passed,
                total_passed,
                failed_checks,
                len(warnings),
            )
            return result

        except Exception as exc:
            logger.exception("run_checklist crashed: {}", exc)
            return {
                "passed": False,
                "score": 0,
                "failed_checks": ["system_error"],
                "warnings": [f"Checklist error: {exc}"],
            }

    # ------------------------------------------------------------------
    # Individual check implementations
    # ------------------------------------------------------------------
    def _check_score(
        self, signal: dict, passed: List[str], failed: List[str]
    ) -> None:
        try:
            if signal.get("score", 0) >= self.cfg["min_score"]:
                passed.append("score_threshold")
            else:
                failed.append("score_threshold")
        except Exception as exc:
            logger.error("_check_score error: {}", exc)
            failed.append("score_threshold")

    def _check_risk_reward(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            entry = signal.get("entry", 0)
            target = signal.get("target", 0)
            sl = signal.get("sl", 0)
            risk = abs(entry - sl)
            reward = abs(target - entry)

            if risk <= 0:
                failed.append("risk_reward_ratio")
                warnings.append("Risk is zero or negative; cannot compute R:R")
                return

            rr = reward / risk
            if rr >= self.cfg["min_rr_ratio"]:
                passed.append("risk_reward_ratio")
            else:
                failed.append("risk_reward_ratio")
                warnings.append(f"R:R {rr:.2f} below minimum {self.cfg['min_rr_ratio']}")
        except Exception as exc:
            logger.error("_check_risk_reward error: {}", exc)
            failed.append("risk_reward_ratio")

    def _check_iv_rank(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            iv_rank = signal.get("iv_rank", 0)
            max_iv = self.cfg["max_iv_rank"]
            if iv_rank < max_iv:
                passed.append("iv_rank_acceptable")
            else:
                failed.append("iv_rank_acceptable")
                warnings.append(f"IV rank {iv_rank:.1f} >= {max_iv} (expensive options)")
        except Exception as exc:
            logger.error("_check_iv_rank error: {}", exc)
            failed.append("iv_rank_acceptable")

    def _check_upcoming_events(
        self,
        market_state: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            now_ist = datetime.now(IST)
            lookahead = timedelta(hours=self.cfg["event_lookahead_hours"])
            events = market_state.get("upcoming_events", [])

            event_near = False
            for evt in events:
                evt_time = evt.get("time")
                if evt_time is None:
                    continue
                if isinstance(evt_time, str):
                    evt_time = datetime.fromisoformat(evt_time)
                if not evt_time.tzinfo:
                    evt_time = IST.localize(evt_time)
                if now_ist <= evt_time <= now_ist + lookahead:
                    event_near = True
                    warnings.append(
                        f"Event '{evt.get('name', 'unknown')}' at {evt_time.strftime('%H:%M IST')}"
                    )
                    break

            if not event_near:
                passed.append("no_nearby_event")
            else:
                failed.append("no_nearby_event")
        except Exception as exc:
            logger.error("_check_upcoming_events error: {}", exc)
            failed.append("no_nearby_event")

    def _check_vix(
        self,
        market_state: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            vix = market_state.get("vix", 0)
            max_vix = self.cfg["max_vix"]
            vix_warn = self.cfg["vix_warn"]

            if vix > max_vix:
                failed.append("vix_acceptable")
                warnings.append(f"VIX {vix:.2f} exceeds max {max_vix}")
            else:
                passed.append("vix_acceptable")
                if vix >= vix_warn:
                    warnings.append(f"VIX {vix:.2f} in caution zone ({vix_warn}-{max_vix})")
        except Exception as exc:
            logger.error("_check_vix error: {}", exc)
            failed.append("vix_acceptable")

    def _check_spread(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            bid = signal.get("bid", 0)
            ask = signal.get("ask", 0)
            premium = signal.get("premium", 0)

            if premium <= 0:
                failed.append("spread_acceptable")
                warnings.append("Premium is zero/negative; cannot evaluate spread")
                return

            spread = abs(ask - bid)
            spread_pct = (spread / premium) * 100.0
            max_spread = self.cfg["max_spread_pct"]

            if spread_pct < max_spread:
                passed.append("spread_acceptable")
            else:
                failed.append("spread_acceptable")
                warnings.append(f"Bid-ask spread {spread_pct:.1f}% exceeds {max_spread}% of premium")
        except Exception as exc:
            logger.error("_check_spread error: {}", exc)
            failed.append("spread_acceptable")

    def _check_volume(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            volume = signal.get("volume", 0)
            min_vol = self.cfg["min_volume"]

            if volume >= min_vol:
                passed.append("volume_adequate")
            else:
                failed.append("volume_adequate")
                warnings.append(f"Volume {volume} below minimum {min_vol}")
        except Exception as exc:
            logger.error("_check_volume error: {}", exc)
            failed.append("volume_adequate")

    def _check_time_to_expiry(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
        warnings: List[str],
    ) -> None:
        try:
            now_ist = datetime.now(IST)
            expiry = signal.get("expiry")

            if expiry is None:
                failed.append("time_to_expiry")
                warnings.append("Expiry not provided in signal")
                return

            if isinstance(expiry, str):
                expiry = datetime.fromisoformat(expiry)
            if not expiry.tzinfo:
                expiry = IST.localize(expiry)

            hours_left = (expiry - now_ist).total_seconds() / 3600.0
            min_hours = self.cfg["min_hours_to_expiry"]

            if hours_left >= min_hours:
                passed.append("time_to_expiry")
            else:
                failed.append("time_to_expiry")
                warnings.append(f"Only {hours_left:.1f}h to expiry (min {min_hours}h)")
        except Exception as exc:
            logger.error("_check_time_to_expiry error: {}", exc)
            failed.append("time_to_expiry")

    def _check_circuit_breaker(
        self,
        market_state: dict,
        passed: List[str],
        failed: List[str],
    ) -> None:
        try:
            if market_state.get("circuit_breaker_active", False):
                failed.append("no_circuit_breaker")
            else:
                passed.append("no_circuit_breaker")
        except Exception as exc:
            logger.error("_check_circuit_breaker error: {}", exc)
            failed.append("no_circuit_breaker")

    def _check_capital_allocation(
        self,
        signal: dict,
        passed: List[str],
        failed: List[str],
    ) -> None:
        try:
            alloc_pct = signal.get("capital_allocation_pct", 0)
            max_alloc = self.cfg["max_capital_alloc_pct"]

            if alloc_pct <= max_alloc:
                passed.append("capital_allocation")
            else:
                failed.append("capital_allocation")
        except Exception as exc:
            logger.error("_check_capital_allocation error: {}", exc)
            failed.append("capital_allocation")
