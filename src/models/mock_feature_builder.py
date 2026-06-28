"""
Feature Builder for Mock Dataset
Reads synthetic match data and generates ML features.
Calculates rolling averages, form differences, and implicit probabilities.
"""

from pathlib import Path

import pandas as pd

from src.utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR
from src.utils.logger import logger


class MockFeatureBuilder:
    """
    Builds ML features from mock match dataset.
    Calculates rolling averages and form metrics for training.
    """

    def __init__(self, input_file: str = "mock_dataset.csv"):
        self.input_path = DATA_RAW_DIR / "historical" / input_file
        self.output_dir = DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load mock dataset from CSV."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Mock dataset not found at {self.input_path}")

        df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(df)} matches from {self.input_path}")
        return df

    def _calculate_rolling_stats(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Calculate rolling averages for each team.
        Adds columns for goals scored/conceded in last N matches.
        """
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        # Initialize columns
        df["home_goals_for_rolling"] = 0.0
        df["home_goals_against_rolling"] = 0.0
        df["away_goals_for_rolling"] = 0.0
        df["away_goals_against_rolling"] = 0.0

        # Track team history
        team_history: dict[str, list[dict[str, int]]] = {}

        for idx, row in df.iterrows():
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            home_goals = int(row["home_goals"])  # type: ignore[arg-type]
            away_goals = int(row["away_goals"])  # type: ignore[arg-type]

            # Get home team rolling stats
            if home_team in team_history and len(team_history[home_team]) >= window:
                recent_matches = team_history[home_team][-window:]
                home_gf = sum(m["goals_for"] for m in recent_matches) / window
                home_ga = sum(m["goals_against"] for m in recent_matches) / window
            else:
                home_gf = 1.45  # Default to league average
                home_ga = 1.15

            # Get away team rolling stats
            if away_team in team_history and len(team_history[away_team]) >= window:
                recent_matches = team_history[away_team][-window:]
                away_gf = sum(m["goals_for"] for m in recent_matches) / window
                away_ga = sum(m["goals_against"] for m in recent_matches) / window
            else:
                away_gf = 1.15
                away_ga = 1.45

            # Update DataFrame
            df.at[idx, "home_goals_for_rolling"] = home_gf
            df.at[idx, "home_goals_against_rolling"] = home_ga
            df.at[idx, "away_goals_for_rolling"] = away_gf
            df.at[idx, "away_goals_against_rolling"] = away_ga

            # Update team history
            if home_team not in team_history:
                team_history[home_team] = []
            team_history[home_team].append(
                {
                    "goals_for": home_goals,
                    "goals_against": away_goals,
                }
            )

            if away_team not in team_history:
                team_history[away_team] = []
            team_history[away_team].append(
                {
                    "goals_for": away_goals,
                    "goals_against": home_goals,
                }
            )

        return df

    def _calculate_form_difference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate difference in recent form between home and away teams."""
        df["form_diff"] = (df["home_goals_for_rolling"] - df["home_goals_against_rolling"]) - (
            df["away_goals_for_rolling"] - df["away_goals_against_rolling"]
        )
        return df

    def _calculate_implicit_probability(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate implicit probability from closing odds."""
        df = df.copy()
        df["home_win_prob"] = 1 / df["BSH"]
        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable: home_win (1 if home team wins, 0 otherwise)."""
        df = df.copy()
        df["home_win"] = (df["home_goals"] > df["away_goals"]).astype(int)
        return df

    def build_features(self) -> pd.DataFrame:
        """
        Build complete feature set for ML training.
        Returns DataFrame with all features and target.
        """
        logger.info("Starting feature engineering")

        # Load data
        df = self.load_data()

        # Calculate rolling stats
        df = self._calculate_rolling_stats(df, window=5)

        # Calculate form difference
        df = self._calculate_form_difference(df)

        # Calculate implicit probability
        df = self._calculate_implicit_probability(df)

        # Create target
        df = self._create_target(df)

        # Select features for training
        feature_columns = [
            "home_goals_for_rolling",
            "home_goals_against_rolling",
            "away_goals_for_rolling",
            "away_goals_against_rolling",
            "form_diff",
            "home_win_prob",
            "home_win",
        ]

        df_features = df[feature_columns].copy()

        # Remove any rows with NaN
        df_features = df_features.dropna()

        logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
        return pd.DataFrame(df_features)

    def save_to_parquet(self, df: pd.DataFrame, filename: str = "training_dataset.parquet") -> Path:
        """Save DataFrame to parquet file."""
        output_path = self.output_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved training dataset to {output_path}")
        return output_path

    def run(self) -> Path:
        """
        Run full feature engineering pipeline.
        Returns path to generated parquet file.
        """
        df = self.build_features()
        output_path = self.save_to_parquet(df)
        return output_path


def main():
    """Run feature builder."""
    builder = MockFeatureBuilder()
    output_path = builder.run()
    print(f"Training dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
