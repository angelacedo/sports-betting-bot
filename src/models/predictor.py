"""
XGBoost Predictor Module
Trains XGBoost models for 1X2 and Over/Under 2.5 markets.
Uses RandomizedSearchCV for hyperparameter tuning.
Validates with cross_val_score. Saves models to /models/.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import polars as pl
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split

from src.utils.config import DATA_PROCESSED_DIR, PROJECT_ROOT
from src.utils.logger import logger

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS: list[str] = [
    "home_score",
    "away_score",
    "goal_diff",
    "home_goals_for_5",
    "home_goals_against_5",
    "away_goals_for_5",
    "away_goals_against_5",
    "home_goals_for_10",
    "home_goals_against_10",
    "away_goals_for_10",
    "away_goals_against_10",
    "goal_form_diff_5",
    "goal_form_diff_10",
    "home_goal_net_5",
    "away_goal_net_5",
    "home_goal_net_10",
    "away_goal_net_10",
    "home_elo",
    "away_elo",
    "elo_diff",
]

XG_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 300, 500],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
}


class Predictor:
    """XGBoost predictor for 1X2 and Over/Under 2.5 markets."""

    def __init__(self) -> None:
        self.model_1x2: xgb.XGBClassifier | None = None
        self.model_ou: xgb.XGBClassifier | None = None
        self.metrics: dict[str, Any] = {}

    def _load_dataset(self, parquet_path: Path | None = None) -> pl.DataFrame:
        """Load training dataset from parquet."""
        if parquet_path is None:
            parquet_path = DATA_PROCESSED_DIR / "training_dataset.parquet"

        if not parquet_path.exists():
            logger.error(f"Dataset not found: {parquet_path}")
            return pl.DataFrame()

        df = pl.read_parquet(parquet_path)
        logger.info(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
        return df

    def _prepare_features(self, df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
        """Select and validate feature columns."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns: {missing}")
        if not available:
            logger.error("No feature columns available")
            return pl.DataFrame(), []
        return df.select(available), available

    def train(self, parquet_path: Path | None = None) -> dict[str, Any]:
        """Train both models (1X2 and Over/Under 2.5)."""
        logger.info("Starting XGBoost training")

        df = self._load_dataset(parquet_path)
        if df.is_empty():
            raise ValueError("Empty dataset")

        features_df, feature_cols = self._prepare_features(df)
        if features_df.is_empty():
            raise ValueError("No features available")

        pdf = features_df.to_pandas()
        y_1x2 = df["match_result"].to_pandas()
        y_ou = df["over_25"].to_pandas()

        x_train, x_test, y_train_1x2, y_test_1x2 = train_test_split(
            pdf, y_1x2, test_size=0.2, random_state=42, stratify=y_1x2
        )
        _, _, y_train_ou, y_test_ou = train_test_split(
            pdf, y_ou, test_size=0.2, random_state=42, stratify=y_ou
        )

        self.model_1x2 = self._train_model(x_train, y_train_1x2, x_test, y_test_1x2, "1x2")
        self.model_ou = self._train_model(x_train, y_train_ou, x_test, y_test_ou, "over_under")

        self.metrics["feature_columns"] = feature_cols
        self.metrics["train_size"] = len(x_train)
        self.metrics["test_size"] = len(x_test)

        self._save_models()
        self._save_metrics()

        logger.info(
            f"Training complete. Metrics: {json.dumps(self.metrics, indent=2, default=str)}"
        )
        return self.metrics

    def _train_model(
        self,
        x_train: Any,
        y_train: Any,
        x_test: Any,
        y_test: Any,
        model_name: str,
    ) -> xgb.XGBClassifier:
        """Train single XGBoost model with RandomizedSearchCV."""
        logger.info(f"Training {model_name} model")

        base_model = xgb.XGBClassifier(
            objective="multi:softprob" if model_name == "1x2" else "binary:logistic",
            eval_metric="mlogloss" if model_name == "1x2" else "logloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )

        search = RandomizedSearchCV(
            base_model,
            param_distributions=XG_PARAM_GRID,
            n_iter=20,
            cv=3,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(x_train, y_train)

        best_model = search.best_estimator_
        test_score = best_model.score(x_test, y_test)

        cv_scores = cross_val_score(best_model, x_train, y_train, cv=5, scoring="accuracy")

        self.metrics[model_name] = {
            "best_params": search.best_params_,
            "test_accuracy": round(test_score, 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
        }

        logger.info(
            f"{model_name}: test_acc={test_score:.4f}, "
            f"cv={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )
        return best_model

    def _save_models(self) -> None:
        """Save trained models to disk."""
        if self.model_1x2:
            path = MODELS_DIR / "xgboost_v1.joblib"
            joblib.dump({"model_1x2": self.model_1x2, "model_ou": self.model_ou}, path)
            logger.info(f"Models saved to {path}")

    def _save_metrics(self) -> None:
        """Save training metrics to JSON."""
        path = MODELS_DIR / "xgboost_metrics.json"
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {path}")

    def load_models(self) -> None:
        """Load trained models from disk."""
        path = MODELS_DIR / "xgboost_v1.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Models not found: {path}")

        data = joblib.load(path)
        self.model_1x2 = data["model_1x2"]
        self.model_ou = data["model_ou"]
        logger.info(f"Models loaded from {path}")

    def predict_1x2(self, features: dict[str, float]) -> dict[str, float]:
        """Predict 1X2 probabilities."""
        if not self.model_1x2:
            raise ValueError("Model not loaded. Call load_models() first.")

        feat_cols = self.metrics.get("feature_columns", FEATURE_COLS)
        x = [[features.get(c, 0.0) for c in feat_cols]]
        probs = self.model_1x2.predict_proba(x)[0]
        classes = self.model_1x2.classes_

        return {str(cls): float(prob) for cls, prob in zip(classes, probs)}

    def predict_over_under(self, features: dict[str, float]) -> float:
        """Predict Over 2.5 probability."""
        if not self.model_ou:
            raise ValueError("Model not loaded. Call load_models() first.")

        feat_cols = self.metrics.get("feature_columns", FEATURE_COLS)
        x = [[features.get(c, 0.0) for c in feat_cols]]
        prob = self.model_ou.predict_proba(x)[0]

        return float(prob[1])


def main() -> None:
    """Train models."""
    predictor = Predictor()
    metrics = predictor.train()
    logger.info(f"Training metrics: {metrics}")


if __name__ == "__main__":
    main()
