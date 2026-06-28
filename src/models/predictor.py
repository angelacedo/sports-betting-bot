"""
XGBoost Predictor
Trains XGBoost model on synthetic match data and evaluates performance.
Saves trained model for future predictions.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils.config import DATA_PROCESSED_DIR
from src.utils.logger import logger


class MatchPredictor:
    """
    Trains and evaluates XGBoost model for match outcome prediction.
    Predicts home_win (1 if home team wins, 0 otherwise).
    """

    def __init__(self, input_file: str = "training_dataset.parquet"):
        self.input_path = DATA_PROCESSED_DIR / input_file
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []

    def load_data(self) -> pd.DataFrame:
        """Load training dataset from parquet."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Training dataset not found at {self.input_path}")

        df = pd.read_parquet(self.input_path)
        logger.info(f"Loaded training dataset: {df.shape}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target.
        Returns (features, target) tuple.
        """
        target_col = "home_win"
        feature_cols = [col for col in df.columns if col != target_col]

        features = df[feature_cols].copy()
        target = df[target_col].copy()

        self.feature_columns = feature_cols
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(features)}")
        return features, target

    def train_model(self, features: pd.DataFrame, target: pd.Series) -> XGBClassifier:
        """
        Train XGBoost classifier.
        Returns trained model.
        """
        logger.info("Training XGBoost model")

        # Split data
        features_train, features_test, target_train, target_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )

        logger.info(f"Train size: {len(features_train)}, Test size: {len(features_test)}")

        # Initialize model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

        # Train
        model.fit(features_train, target_train)

        # Evaluate
        target_pred = model.predict(features_test)
        target_pred_proba = model.predict_proba(features_test)[:, 1]

        accuracy = accuracy_score(target_test, target_pred)
        roc_auc = roc_auc_score(target_test, target_pred_proba)

        logger.info(f"Model trained - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        print("\n=== Model Performance ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC:  {roc_auc:.4f}")
        print("========================\n")

        self.model = model
        return model

    def save_model(self, filename: str = "xgboost_v1.joblib") -> Path:
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        output_path = self.model_dir / filename
        joblib.dump(self.model, output_path)
        logger.info(f"Saved model to {output_path}")
        return output_path

    def run(self) -> Path:
        """
        Run full training pipeline.
        Returns path to saved model.
        """
        # Load data
        df = self.load_data()

        # Prepare features
        features, target = self.prepare_features(df)

        # Train model
        self.train_model(features, target)

        # Save model
        output_path = self.save_model()
        return output_path


def main():
    """Train and save model."""
    predictor = MatchPredictor()
    model_path = predictor.run()
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
