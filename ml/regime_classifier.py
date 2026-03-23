"""
AstroNifty Regime Classifier
Classifies market regime as BULL / BEAR / SIDEWAYS using RandomForest
trained on technical indicators, OI data, and astro scores.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from loguru import logger

REGIME_LABELS = {0: "BEAR", 1: "SIDEWAYS", 2: "BULL"}
REGIME_CODES = {"BEAR": 0, "SIDEWAYS": 1, "BULL": 2}
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "regime_model.joblib"
SCALER_PATH = MODEL_DIR / "regime_scaler.joblib"

FEATURE_COLS = [
    "EMA_9", "EMA_21", "EMA_50", "RSI_14", "ATR_14",
    "volume_ratio", "OI_change", "PCR", "astro_score",
]


class RegimeClassifier:
    """Predicts market regime (BULL / BEAR / SIDEWAYS) from a feature vector."""

    def __init__(self):
        self.model: RandomForestClassifier | None = None
        self.scaler: StandardScaler | None = None
        self._ensure_model_dir()
        if MODEL_PATH.exists():
            self.load_model()
            logger.info("Loaded existing regime model from {}", MODEL_PATH)
        else:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.scaler = StandardScaler()
            logger.info("Initialised fresh RandomForestClassifier for regime detection")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        historical_df: pd.DataFrame,
        oi_data: dict | None = None,
        astro_data: dict | None = None,
    ) -> pd.DataFrame:
        """Build the feature matrix from raw OHLCV + optional OI / astro data.

        Parameters
        ----------
        historical_df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
        oi_data : dict, optional
            Keys expected: ``oi_change``, ``pcr``.
        astro_data : dict, optional
            Key expected: ``astro_score`` (float 0-1).

        Returns
        -------
        pd.DataFrame  with columns matching ``FEATURE_COLS``.
        """
        df = historical_df.copy()
        df.columns = [c.lower() for c in df.columns]

        # --- EMAs ---
        df["EMA_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["EMA_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["EMA_50"] = df["close"].ewm(span=50, adjust=False).mean()

        # --- RSI 14 ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

        # --- ATR 14 ---
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - df["close"].shift()).abs(),
                (df["low"] - df["close"].shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["ATR_14"] = tr.ewm(span=14, adjust=False).mean()

        # --- Volume ratio (current / 20-day SMA) ---
        vol_sma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

        # --- OI fields ---
        if oi_data is not None:
            df["OI_change"] = oi_data.get("oi_change", 0.0)
            df["PCR"] = oi_data.get("pcr", 1.0)
        else:
            df["OI_change"] = 0.0
            df["PCR"] = 1.0

        # --- Astro score ---
        if astro_data is not None:
            df["astro_score"] = astro_data.get("astro_score", 0.5)
        else:
            df["astro_score"] = 0.5

        # Normalise EMA distances relative to close
        for ema in ("EMA_9", "EMA_21", "EMA_50"):
            df[ema] = (df["close"] - df[ema]) / df["close"] * 100.0

        df = df[FEATURE_COLS].dropna().reset_index(drop=True)
        logger.debug("Prepared {} rows x {} features", len(df), len(FEATURE_COLS))
        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> dict:
        """Return regime prediction with class probabilities.

        Parameters
        ----------
        features : pd.DataFrame
            Single row (or last row used) with ``FEATURE_COLS``.

        Returns
        -------
        dict with keys: regime, bull_prob, bear_prob, sideways_prob, confidence
        """
        if self.model is None:
            logger.warning("Model not loaded; returning neutral prediction")
            return {
                "regime": "SIDEWAYS",
                "bull_prob": 0.33,
                "bear_prob": 0.33,
                "sideways_prob": 0.34,
                "confidence": 0.0,
            }

        row = features.iloc[[-1]][FEATURE_COLS].values
        if self.scaler is not None:
            row = self.scaler.transform(row)

        pred = self.model.predict(row)[0]
        probs = self.model.predict_proba(row)[0]

        # Map index -> label.  Model classes may be [0,1,2] or subset.
        prob_map = {REGIME_LABELS.get(c, "SIDEWAYS"): float(p) for c, p in zip(self.model.classes_, probs)}

        bull_p = prob_map.get("BULL", 0.0)
        bear_p = prob_map.get("BEAR", 0.0)
        side_p = prob_map.get("SIDEWAYS", 0.0)
        confidence = float(max(probs))

        result = {
            "regime": REGIME_LABELS.get(int(pred), "SIDEWAYS"),
            "bull_prob": round(bull_p, 4),
            "bear_prob": round(bear_p, 4),
            "sideways_prob": round(side_p, 4),
            "confidence": round(confidence, 4),
        }
        logger.info("Regime prediction: {} (conf {:.1%})", result["regime"], confidence)
        return result

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the RandomForest with 5-fold stratified cross-validation.

        Parameters
        ----------
        X : pd.DataFrame  feature matrix (columns = FEATURE_COLS).
        y : pd.Series      integer-encoded labels (0=BEAR, 1=SIDEWAYS, 2=BULL).

        Returns
        -------
        dict with accuracy, f1_score, classification_report.
        """
        logger.info("Training regime classifier on {} samples", len(X))

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X[FEATURE_COLS])

        # Cross-validation scores
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="f1_weighted")

        # Final fit on full data
        self.model.fit(X_scaled, y)

        y_pred = self.model.predict(X_scaled)
        report = classification_report(y, y_pred, target_names=["BEAR", "SIDEWAYS", "BULL"], output_dict=True)

        metrics = {
            "accuracy": round(float(np.mean(cv_acc)), 4),
            "accuracy_std": round(float(np.std(cv_acc)), 4),
            "f1_score": round(float(np.mean(cv_f1)), 4),
            "f1_std": round(float(np.std(cv_f1)), 4),
            "classification_report": report,
        }
        logger.info(
            "Training complete  |  CV accuracy={:.2%} +/-{:.2%}  |  CV F1={:.2%}",
            metrics["accuracy"],
            metrics["accuracy_std"],
            metrics["f1_score"],
        )
        self.save_model()
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self) -> None:
        """Persist model + scaler to disk."""
        self._ensure_model_dir()
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.scaler, SCALER_PATH)
        logger.info("Saved regime model to {}", MODEL_PATH)

    def load_model(self) -> None:
        """Load model + scaler from disk."""
        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            logger.info("Loaded regime model from {}", MODEL_PATH)
        else:
            logger.warning("No saved model found at {}", MODEL_PATH)

        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
        else:
            self.scaler = StandardScaler()
            logger.warning("No saved scaler found; created fresh StandardScaler")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_model_dir() -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
