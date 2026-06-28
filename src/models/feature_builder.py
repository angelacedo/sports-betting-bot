"""
Feature Builder Module
Generates ML features from historical match data stored in PostgreSQL.
Uses Polars for fast vectorized operations and creates training dataset.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, League, Match, MatchStat, OddsHistory, Team
from src.utils.config import DATA_PROCESSED_DIR, DATABASE_URL
from src.utils.logger import logger


class FeatureBuilder:
    """
    Builds ML features from historical match data.
    Calculates rolling averages, Elo ratings, fatigue metrics, and odds movements.
    """

    def __init__(self, db_url: str = DATABASE_URL):
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.output_dir = DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_matches_from_db(self) -> pl.DataFrame:
        """Fetch all historical matches from PostgreSQL."""
        query = """
        SELECT
            m.id,
            m.external_id,
            m.kickoff,
            m.status,
            m.home_score,
            m.away_score,
            m.season,
            l.name as league_name,
            ht.name as home_team_name,
            at.name as away_team_name
        FROM matches m
        JOIN leagues l ON m.league_id = l.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.status = 'finished'
        ORDER BY m.kickoff ASC
        """

        try:
            with self.engine.connect() as conn:
                df = pl.read_sql(query, conn)
                logger.info(f"Fetched {len(df)} matches from database")
                return df

        except Exception as e:
            logger.error(f"Failed to fetch matches: {e}")
            return pl.DataFrame()

    def _calculate_match_result(self, home_score: int, away_score: int) -> str:
        """Determine match result (H/D/A)."""
        if home_score > away_score:
            return "H"
        elif home_score < away_score:
            return "A"
        else:
            return "D"

    def _calculate_total_goals(self, home_score: int, away_score: int) -> int:
        """Calculate total goals in match."""
        return home_score + away_score

    def _calculate_rolling_averages(
        self,
        df: pl.DataFrame,
        team_col: str,
        score_col: str,
        windows: list[int] | None = None,
    ) -> pl.DataFrame:
        """
        Calculate rolling averages for a team statistic.
        Returns DataFrame with rolling average columns.
        """
        if windows is None:
            windows = [5, 10]

        result_df = df.clone()

        for window in windows:
            col_name = f"{score_col}_rolling_{window}"

            # Group by team and calculate rolling mean
            rolling_stats = df.group_by(team_col).agg(
                [pl.col(score_col).rolling_mean(window_size=window).alias(col_name)]
            )

            # Merge back to main DataFrame
            result_df = result_df.join(
                rolling_stats,
                left_on=team_col,
                right_on=team_col,
                how="left",
            )

        return result_df

    def _calculate_elo_ratings(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate Elo ratings for each team based on match results.
        Uses standard Elo formula with K-factor of 20.
        """
        # Initialize Elo ratings
        teams = set(df["home_team_name"].to_list() + df["away_team_name"].to_list())
        elo_ratings = {team: 1500.0 for team in teams}

        K_FACTOR = 20
        elo_history = []

        for row in df.iter_rows(named=True):
            home_team = row["home_team_name"]
            away_team = row["away_team_name"]
            home_score = row["home_score"]
            away_score = row["away_score"]

            # Get current ratings
            home_elo = elo_ratings[home_team]
            away_elo = elo_ratings[away_team]

            # Calculate expected scores
            expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
            expected_away = 1 - expected_home

            # Determine actual result (1=win, 0.5=draw, 0=loss)
            if home_score > away_score:
                actual_home, actual_away = 1.0, 0.0
            elif home_score < away_score:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5

            # Update ratings
            new_home_elo = home_elo + K_FACTOR * (actual_home - expected_home)
            new_away_elo = away_elo + K_FACTOR * (actual_away - expected_away)

            elo_ratings[home_team] = new_home_elo
            elo_ratings[away_team] = new_away_elo

            # Store for this match
            elo_history.append(
                {
                    "id": row["id"],
                    "home_elo_before": home_elo,
                    "away_elo_before": away_elo,
                    "elo_diff": home_elo - away_elo,
                }
            )

        # Convert to DataFrame and merge
        elo_df = pl.DataFrame(elo_history)
        return df.join(elo_df, on="id", how="left")

    def _calculate_fatigue(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate days of rest between matches for each team.
        Lower rest = higher fatigue.
        """
        result_df = df.clone()

        # Sort by team and date
        home_matches = df.select(
            [
                pl.col("home_team_name").alias("team"),
                pl.col("kickoff").alias("match_date"),
                pl.col("id"),
            ]
        )

        away_matches = df.select(
            [
                pl.col("away_team_name").alias("team"),
                pl.col("kickoff").alias("match_date"),
                pl.col("id"),
            ]
        )

        all_matches = pl.concat([home_matches, away_matches])
        all_matches = all_matches.sort(["team", "match_date"])

        # Calculate days since last match
        fatigue_data = []
        for team in all_matches["team"].unique().to_list():
            team_matches = all_matches.filter(pl.col("team") == team)

            if len(team_matches) < 2:
                continue

            dates = team_matches["match_date"].to_list()
            match_ids = team_matches["id"].to_list()

            for i in range(1, len(dates)):
                days_rest = (dates[i] - dates[i - 1]).days
                fatigue_data.append(
                    {
                        "id": match_ids[i],
                        "days_rest": days_rest,
                    }
                )

        if fatigue_data:
            fatigue_df = pl.DataFrame(fatigue_data)
            result_df = result_df.join(fatigue_df, on="id", how="left")
        else:
            result_df = result_df.with_columns(pl.lit(None).alias("days_rest"))

        return result_df

    def _calculate_home_away_performance(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate home/away performance metrics.
        Home teams typically have advantage.
        """
        result_df = df.clone()

        # Calculate home team performance (last 5 home matches)
        home_perf = df.group_by("home_team_name").agg(
            [
                pl.col("home_score").rolling_mean(window_size=5).alias("home_team_form"),
                (pl.col("home_score") > pl.col("away_score"))
                .cast(pl.Int32)
                .rolling_mean(window_size=5)
                .alias("home_win_rate"),
            ]
        )

        # Calculate away team performance (last 5 away matches)
        away_perf = df.group_by("away_team_name").agg(
            [
                pl.col("away_score").rolling_mean(window_size=5).alias("away_team_form"),
                (pl.col("away_score") > pl.col("home_score"))
                .cast(pl.Int32)
                .rolling_mean(window_size=5)
                .alias("away_win_rate"),
            ]
        )

        # Merge
        result_df = result_df.join(
            home_perf,
            left_on="home_team_name",
            right_on="home_team_name",
            how="left",
        )

        result_df = result_df.join(
            away_perf,
            left_on="away_team_name",
            right_on="away_team_name",
            how="left",
        )

        return result_df

    def build_features(self) -> pl.DataFrame:
        """
        Build complete feature set for ML training.
        Returns Polars DataFrame with all features and targets.
        """
        logger.info("Starting feature engineering")

        # Fetch data
        df = self._fetch_matches_from_db()
        if df.is_empty():
            logger.warning("No matches found in database")
            return pl.DataFrame()

        initial_count = len(df)
        logger.info(f"Processing {initial_count} matches")

        # Calculate targets
        df = df.with_columns(
            [
                pl.struct(["home_score", "away_score"])
                .map_elements(
                    lambda x: self._calculate_match_result(x["home_score"], x["away_score"])
                )
                .alias("match_result"),
                pl.struct(["home_score", "away_score"])
                .map_elements(
                    lambda x: self._calculate_total_goals(x["home_score"], x["away_score"])
                )
                .alias("total_goals"),
            ]
        )

        # Calculate Elo ratings
        logger.info("Calculating Elo ratings")
        df = self._calculate_elo_ratings(df)

        # Calculate fatigue
        logger.info("Calculating fatigue metrics")
        df = self._calculate_fatigue(df)

        # Calculate home/away performance
        logger.info("Calculating home/away performance")
        df = self._calculate_home_away_performance(df)

        # Calculate rolling averages for goals
        logger.info("Calculating rolling averages")
        df = self._calculate_rolling_averages(df, "home_team_name", "home_score")
        df = self._calculate_rolling_averages(df, "away_team_name", "away_score")

        # Remove rows with null values (from rolling windows)
        df = df.drop_nulls(subset=["home_elo_before", "away_elo_before"])

        final_count = len(df)
        logger.info(f"Feature engineering complete. {final_count}/{initial_count} matches retained")

        return df

    def save_to_parquet(self, df: pl.DataFrame, filename: str = "training_dataset.parquet") -> Path:
        """Save DataFrame to parquet file."""
        output_path = self.output_dir / filename
        df.write_parquet(output_path)
        logger.info(f"Saved training dataset to {output_path}")
        return output_path

    def run(self) -> Path:
        """
        Run full feature engineering pipeline.
        Returns path to generated parquet file.
        """
        df = self.build_features()

        if df.is_empty():
            logger.error("No features generated")
            raise ValueError("Feature engineering failed")

        output_path = self.save_to_parquet(df)
        logger.info(f"Feature engineering complete. Dataset saved to {output_path}")

        return output_path


def main():
    """Run feature builder."""
    builder = FeatureBuilder()
    output_path = builder.run()
    print(f"Training dataset saved to: {output_path}")


if __name__ == "__main__":
    main()
