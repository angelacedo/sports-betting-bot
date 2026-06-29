"""
Feature Builder Module
Generates ML features from historical match data stored in PostgreSQL.
Uses Polars for fast vectorized operations.
Features: rolling goals for/against (5, 10), Elo ratings, home/away form.
Targets: match_result (H/D/A), over_25.
"""

from pathlib import Path

import polars as pl
from sqlalchemy import create_engine

from src.utils.config import DATA_PROCESSED_DIR, DATABASE_URL
from src.utils.logger import logger


class FeatureBuilder:
    """
    Builds ML features from historical match data.
    Rolling goals, Elo ratings, home/away performance.
    """

    def __init__(self, db_url: str = DATABASE_URL) -> None:
        self.engine = create_engine(db_url)
        self.output_dir = DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_matches(self) -> pl.DataFrame:
        """Fetch all finished matches from PostgreSQL."""
        query = """
        SELECT
            m.id,
            m.external_id,
            m.kickoff,
            m.home_score,
            m.away_score,
            m.season,
            l.name AS league_name,
            ht.name AS home_team_name,
            at.name AS away_team_name
        FROM matches m
        JOIN leagues l ON m.league_id = l.id
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        WHERE m.status = 'finished'
          AND m.home_score IS NOT NULL
          AND m.away_score IS NOT NULL
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

    def _add_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add target columns: match_result and over_25."""
        return df.with_columns(
            [
                pl.when(pl.col("home_score") > pl.col("away_score"))
                .then(pl.lit("H"))
                .when(pl.col("home_score") < pl.col("away_score"))
                .then(pl.lit("A"))
                .otherwise(pl.lit("D"))
                .alias("match_result"),
                ((pl.col("home_score") + pl.col("away_score")) > 2.5)
                .cast(pl.Int8)
                .alias("over_25"),
            ]
        )

    def _rolling_goals(self, df: pl.DataFrame, window: int) -> pl.DataFrame:
        """Calculate rolling goals for/against for each team."""
        home_goals = df.select(
            [
                pl.col("home_team_name").alias("team"),
                pl.col("kickoff"),
                pl.col("home_score").alias("goals_for"),
                pl.col("away_score").alias("goals_against"),
                pl.col("id"),
            ]
        )
        away_goals = df.select(
            [
                pl.col("away_team_name").alias("team"),
                pl.col("kickoff"),
                pl.col("away_score").alias("goals_for"),
                pl.col("home_score").alias("goals_against"),
                pl.col("id"),
            ]
        )
        all_goals = pl.concat([home_goals, away_goals]).sort(["team", "kickoff"])

        rolling = all_goals.with_columns(
            [
                pl.col("goals_for")
                .rolling_mean(window_size=window)
                .alias(f"goals_for_rolling_{window}"),
                pl.col("goals_against")
                .rolling_mean(window_size=window)
                .alias(f"goals_against_rolling_{window}"),
            ]
        )

        home_rolling = rolling.select(
            [
                pl.col("id"),
                pl.col(f"goals_for_rolling_{window}").alias(f"home_goals_for_{window}"),
                pl.col(f"goals_against_rolling_{window}").alias(f"home_goals_against_{window}"),
            ]
        )

        away_rolling = rolling.select(
            [
                pl.col("id"),
                pl.col(f"goals_for_rolling_{window}").alias(f"away_goals_for_{window}"),
                pl.col(f"goals_against_rolling_{window}").alias(f"away_goals_against_{window}"),
            ]
        )

        return df.join(home_rolling, on="id", how="left").join(away_rolling, on="id", how="left")

    def _add_goal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add goal-based features."""
        result = df.clone()

        for window in [5, 10]:
            result = self._rolling_goals(result, window)

        result = result.with_columns(
            [
                (pl.col("home_score") - pl.col("away_score")).alias("goal_diff"),
            ]
        )

        for window in [5, 10]:
            result = result.with_columns(
                [
                    (pl.col(f"home_goals_for_{window}") - pl.col(f"away_goals_for_{window}")).alias(
                        f"goal_form_diff_{window}"
                    ),
                    (
                        pl.col(f"home_goals_for_{window}") - pl.col(f"home_goals_against_{window}")
                    ).alias(f"home_goal_net_{window}"),
                    (
                        pl.col(f"away_goals_for_{window}") - pl.col(f"away_goals_against_{window}")
                    ).alias(f"away_goal_net_{window}"),
                ]
            )

        return result

    def _calculate_elo(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate Elo ratings before each match."""
        teams = set(df["home_team_name"].to_list() + df["away_team_name"].to_list())
        elo: dict[str, float] = {t: 1500.0 for t in teams}
        k = 20
        records: list[dict] = []

        for row in df.iter_rows(named=True):
            h, a = row["home_team_name"], row["away_team_name"]
            h_elo, a_elo = elo[h], elo[a]
            exp_h = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            exp_a = 1 - exp_h

            hs, as_ = row["home_score"], row["away_score"]
            if hs > as_:
                act_h, act_a = 1.0, 0.0
            elif hs < as_:
                act_h, act_a = 0.0, 1.0
            else:
                act_h, act_a = 0.5, 0.5

            elo[h] = h_elo + k * (act_h - exp_h)
            elo[a] = a_elo + k * (act_a - exp_a)

            records.append(
                {
                    "id": row["id"],
                    "home_elo": h_elo,
                    "away_elo": a_elo,
                    "elo_diff": h_elo - a_elo,
                }
            )

        elo_df = pl.DataFrame(records)
        return df.join(elo_df, on="id", how="left")

    def build_features(self) -> pl.DataFrame:
        """Build complete feature set."""
        logger.info("Starting feature engineering")

        df = self._fetch_matches()
        if df.is_empty():
            logger.warning("No matches found")
            return pl.DataFrame()

        initial = len(df)
        logger.info(f"Processing {initial} matches")

        df = self._add_targets(df)
        df = self._add_goal_features(df)
        df = self._calculate_elo(df)

        df = df.drop_nulls(
            subset=[
                "home_goals_for_5",
                "away_goals_for_5",
                "home_goals_for_10",
                "away_goals_for_10",
            ]
        )

        logger.info(f"Features built: {len(df)}/{initial} matches retained")
        return df

    def save_parquet(self, df: pl.DataFrame, filename: str = "training_dataset.parquet") -> Path:
        """Save DataFrame to parquet."""
        path = self.output_dir / filename
        df.write_parquet(path)
        logger.info(f"Saved dataset to {path}")
        return path

    def run(self) -> Path:
        """Full pipeline: build features -> save parquet."""
        df = self.build_features()
        if df.is_empty():
            raise ValueError("Feature engineering produced empty dataset")
        return self.save_parquet(df)


def main() -> None:
    """Run feature builder."""
    builder = FeatureBuilder()
    path = builder.run()
    logger.info(f"Training dataset saved to: {path}")


if __name__ == "__main__":
    main()
