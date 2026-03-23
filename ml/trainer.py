"""
AstroNifty Model Trainer
Collects training data, trains regime + pattern models, runs nightly
retraining, and provides a simple backtesting framework.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from ml.regime_classifier import RegimeClassifier, REGIME_CODES, FEATURE_COLS
from ml.pattern_matcher import PatternMatcher


class ModelTrainer:
    """Orchestrates data collection, training, retraining, and backtesting."""

    def __init__(self, db_manager, kite_client):
        """
        Parameters
        ----------
        db_manager : object
            DB layer -- must expose ``execute_query(sql, params)`` and
            ``get_session()`` context-manager.
        kite_client : object
            Broker client with ``historical_data(instrument_token, from_date,
            to_date, interval)`` returning list[dict].
        """
        self.db = db_manager
        self.kite = kite_client
        self.regime_clf = RegimeClassifier()
        self.pattern_matcher = PatternMatcher(db_manager)
        logger.info("ModelTrainer initialised")

    # ------------------------------------------------------------------
    # 1. Data collection
    # ------------------------------------------------------------------

    def collect_training_data(self, days: int = 90) -> pd.DataFrame:
        """Fetch historical OHLCV, OI snapshots, and astro data and label each
        trading day as BULL / BEAR / SIDEWAYS.

        Labelling rule (close vs open):
        - change > +0.25 %  ->  BULL  (2)
        - change < -0.25 %  ->  BEAR  (0)
        - else               ->  SIDEWAYS (1)

        Returns
        -------
        pd.DataFrame with OHLCV + OI + astro columns + ``label`` column.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # --- OHLCV from broker ---
        logger.info("Fetching OHLCV for last {} days", days)
        try:
            raw = self.kite.historical_data(
                instrument_token=256265,  # NIFTY 50
                from_date=start_date.strftime("%Y-%m-%d"),
                to_date=end_date.strftime("%Y-%m-%d"),
                interval="day",
            )
            ohlcv = pd.DataFrame(raw)
        except Exception as exc:
            logger.warning("Kite historical_data failed ({}), falling back to DB", exc)
            ohlcv = self._ohlcv_from_db(start_date, end_date)

        if ohlcv.empty:
            logger.error("No OHLCV data available for training")
            return pd.DataFrame()

        ohlcv.columns = [c.lower() for c in ohlcv.columns]
        if "date" in ohlcv.columns:
            ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

        # --- OI snapshots from DB ---
        oi_df = self._oi_from_db(start_date, end_date)

        # --- Astro data from DB ---
        astro_df = self._astro_from_db(start_date, end_date)

        # --- Merge ---
        df = ohlcv.copy()
        if not oi_df.empty:
            df = df.merge(oi_df, on="date", how="left")
        else:
            df["oi_change"] = 0.0
            df["pcr"] = 1.0

        if not astro_df.empty:
            df = df.merge(astro_df, on="date", how="left")
        else:
            df["astro_score"] = 0.5

        df.fillna({"oi_change": 0.0, "pcr": 1.0, "astro_score": 0.5}, inplace=True)

        # --- Label ---
        df["change_pct"] = (df["close"] - df["open"]) / df["open"] * 100.0
        df["label"] = df["change_pct"].apply(self._label_from_change)

        logger.info(
            "Collected {} rows  |  BULL={} BEAR={} SIDE={}",
            len(df),
            (df["label"] == 2).sum(),
            (df["label"] == 0).sum(),
            (df["label"] == 1).sum(),
        )
        return df

    # ------------------------------------------------------------------
    # 2. Train regime model
    # ------------------------------------------------------------------

    def train_regime_model(self) -> dict:
        """Prepare features, split 80/20, train RegimeClassifier.

        Returns
        -------
        dict with training metrics (accuracy, f1, report).
        """
        df = self.collect_training_data(days=365)
        if df.empty or len(df) < 50:
            logger.error("Insufficient data to train regime model ({} rows)", len(df))
            return {"error": "insufficient_data", "rows": len(df)}

        features = self.regime_clf.prepare_features(
            df,
            oi_data={"oi_change": 0.0, "pcr": 1.0},
            astro_data={"astro_score": 0.5},
        )

        # Align label column with feature rows (prepare_features drops NaN rows)
        labels = df["label"].iloc[-len(features):].reset_index(drop=True)

        # 80/20 split preserving time order
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = labels.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_test = labels.iloc[split_idx:]

        metrics = self.regime_clf.train(X_train, y_train)

        # Out-of-sample evaluation
        from sklearn.metrics import accuracy_score as acc_fn, f1_score as f1_fn

        X_test_scaled = self.regime_clf.scaler.transform(X_test[FEATURE_COLS])
        y_pred = self.regime_clf.model.predict(X_test_scaled)
        metrics["test_accuracy"] = round(float(acc_fn(y_test, y_pred)), 4)
        metrics["test_f1"] = round(float(f1_fn(y_test, y_pred, average="weighted")), 4)

        logger.info("Regime model test accuracy={:.2%}  test_f1={:.2%}", metrics["test_accuracy"], metrics["test_f1"])
        return metrics

    # ------------------------------------------------------------------
    # 3. Train pattern model
    # ------------------------------------------------------------------

    def train_pattern_model(self) -> dict:
        """Build / refresh the pattern backtest database by matching every
        historical astro configuration with its market outcome.

        Returns
        -------
        dict with patterns_stored count.
        """
        logger.info("Building pattern backtest database")

        query = """
            SELECT ad.date, ad.nakshatra, ad.tithi, ad.hora,
                   md.open, md.close
            FROM astro_data ad
            JOIN market_data md ON ad.date = md.date
            ORDER BY ad.date
        """
        rows = self.db.execute_query(query, {})
        if not rows:
            logger.error("No astro+market data to build pattern DB")
            return {"error": "no_data", "patterns_stored": 0}

        patterns_stored = 0
        for row in rows:
            open_price = float(row.get("open", 0))
            close_price = float(row.get("close", 0))
            if open_price == 0:
                continue

            change_pct = (close_price - open_price) / open_price * 100.0
            if change_pct > 0.25:
                outcome = "bull"
            elif change_pct < -0.25:
                outcome = "bear"
            else:
                outcome = "neutral"

            pattern_name = f"{row.get('nakshatra', '')}_{row.get('tithi', '')}_{row.get('hora', '')}"

            insert_sql = """
                INSERT OR REPLACE INTO pattern_backtest
                    (date, pattern_name, outcome, return_pct)
                VALUES (:date, :pattern_name, :outcome, :return_pct)
            """
            try:
                self.db.execute_query(insert_sql, {
                    "date": row["date"],
                    "pattern_name": pattern_name,
                    "outcome": outcome,
                    "return_pct": round(change_pct, 4),
                })
                patterns_stored += 1
            except Exception as exc:
                logger.warning("Failed to insert pattern row: {}", exc)

        result = {"patterns_stored": patterns_stored}
        logger.info("Pattern backtest DB built: {} records", patterns_stored)
        return result

    # ------------------------------------------------------------------
    # 4. Nightly retrain
    # ------------------------------------------------------------------

    def nightly_retrain(self) -> dict:
        """Run at 11 PM daily. Retrains all models with latest data.

        Returns
        -------
        dict: models_trained (list[str]), metrics (dict per model).
        """
        logger.info("=== Nightly retrain started at {} ===", datetime.utcnow().isoformat())
        results: dict = {"models_trained": [], "metrics": {}}

        # Regime model
        try:
            regime_metrics = self.train_regime_model()
            results["models_trained"].append("regime_classifier")
            results["metrics"]["regime_classifier"] = regime_metrics
        except Exception as exc:
            logger.error("Regime model retrain failed: {}", exc)
            results["metrics"]["regime_classifier"] = {"error": str(exc)}

        # Pattern model
        try:
            pattern_metrics = self.train_pattern_model()
            results["models_trained"].append("pattern_matcher")
            results["metrics"]["pattern_matcher"] = pattern_metrics
        except Exception as exc:
            logger.error("Pattern model retrain failed: {}", exc)
            results["metrics"]["pattern_matcher"] = {"error": str(exc)}

        logger.info(
            "=== Nightly retrain complete  |  models={} ===",
            results["models_trained"],
        )
        return results

    # ------------------------------------------------------------------
    # 5. Backtesting
    # ------------------------------------------------------------------

    def backtest(
        self,
        strategy: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> dict:
        """Simple walk-forward backtest for a given strategy.

        Supported strategies: ``regime_follow`` (buy on BULL signal, sell on
        BEAR signal).

        Returns
        -------
        dict: total_trades, win_rate, avg_return, max_drawdown, sharpe_ratio.
        """
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        logger.info("Running backtest: strategy={}, {} to {}", strategy, start_dt.date(), end_dt.date())

        # Fetch data
        query = """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        rows = self.db.execute_query(query, {
            "start": start_dt.strftime("%Y-%m-%d"),
            "end": end_dt.strftime("%Y-%m-%d"),
        })
        if not rows or len(rows) < 60:
            logger.error("Insufficient market data for backtest ({} rows)", len(rows) if rows else 0)
            return {"error": "insufficient_data"}

        df = pd.DataFrame(rows)
        df.columns = [c.lower() for c in df.columns]

        # Build features day by day (walk-forward)
        trades: list[dict] = []
        position: str | None = None  # "long" or None
        entry_price = 0.0
        peak_equity = 0.0
        equity = 0.0
        max_dd = 0.0

        for i in range(60, len(df)):
            window = df.iloc[: i + 1]
            features = self.regime_clf.prepare_features(window)
            if features.empty:
                continue

            pred = self.regime_clf.predict(features)
            close_today = float(df.iloc[i]["close"])

            if strategy == "regime_follow":
                # Entry
                if position is None and pred["regime"] == "BULL" and pred["confidence"] >= 0.45:
                    position = "long"
                    entry_price = close_today

                # Exit
                elif position == "long" and (pred["regime"] == "BEAR" or pred["confidence"] < 0.35):
                    ret_pct = (close_today - entry_price) / entry_price * 100.0
                    trades.append({
                        "entry_date": str(df.iloc[i - 1]["date"]),
                        "exit_date": str(df.iloc[i]["date"]),
                        "entry_price": entry_price,
                        "exit_price": close_today,
                        "return_pct": round(ret_pct, 4),
                        "win": ret_pct > 0,
                    })
                    equity += ret_pct
                    peak_equity = max(peak_equity, equity)
                    dd = peak_equity - equity
                    max_dd = max(max_dd, dd)
                    position = None

        if not trades:
            logger.warning("Backtest produced zero trades")
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_return": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
            }

        returns = [t["return_pct"] for t in trades]
        wins = sum(1 for t in trades if t["win"])
        avg_ret = float(np.mean(returns))
        std_ret = float(np.std(returns)) if len(returns) > 1 else 1.0
        sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

        result = {
            "total_trades": len(trades),
            "win_rate": round(wins / len(trades), 4),
            "avg_return": round(avg_ret, 4),
            "max_drawdown": round(max_dd, 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "trades": trades,
        }
        logger.info(
            "Backtest done  |  trades={} win_rate={:.1%} avg_ret={:.2%} sharpe={:.2f}",
            result["total_trades"],
            result["win_rate"],
            result["avg_return"],
            result["sharpe_ratio"],
        )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _label_from_change(change_pct: float) -> int:
        if change_pct > 0.25:
            return REGIME_CODES["BULL"]
        elif change_pct < -0.25:
            return REGIME_CODES["BEAR"]
        return REGIME_CODES["SIDEWAYS"]

    def _ohlcv_from_db(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        rows = self.db.execute_query(query, {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _oi_from_db(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = """
            SELECT date, oi_change, pcr
            FROM oi_snapshots
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        rows = self.db.execute_query(query, {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _astro_from_db(self, start: datetime, end: datetime) -> pd.DataFrame:
        query = """
            SELECT date, astro_score
            FROM astro_data
            WHERE date BETWEEN :start AND :end
            ORDER BY date
        """
        rows = self.db.execute_query(query, {
            "start": start.strftime("%Y-%m-%d"),
            "end": end.strftime("%Y-%m-%d"),
        })
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df
