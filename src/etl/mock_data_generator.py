"""
Mock Data Generator
Generates realistic synthetic historical match data for ML pipeline testing.
Uses Poisson distributions for goals and realistic bookmaker odds with vig.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import DATA_RAW_DIR
from src.utils.logger import logger


class MockDataGenerator:
    """
    Generates synthetic football match data with realistic distributions.
    Simulates 3 seasons of a league with ~500 matches per season.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Team pool (20 teams like La Liga)
        self.teams = [
            "Real Madrid",
            "Barcelona",
            "Atletico Madrid",
            "Sevilla",
            "Valencia",
            "Villarreal",
            "Real Sociedad",
            "Athletic Bilbao",
            "Real Betis",
            "Celta Vigo",
            "Getafe",
            "Osasuna",
            "Espanyol",
            "Rayo Vallecano",
            "Mallorca",
            "Girona",
            "Cadiz",
            "Almeria",
            "Valladolid",
            "Elche",
        ]

        # Goal distributions (Poisson lambda)
        self.home_goals_lambda = 1.45
        self.away_goals_lambda = 1.15

        # Shot distributions
        self.home_shots_mean = 14.5
        self.home_shots_std = 4.2
        self.away_shots_mean = 11.8
        self.away_shots_std = 3.8

        # Bookmaker margin (vig)
        self.vig_margin = 0.05

    def _calculate_odds(
        self, home_prob: float, draw_prob: float, away_prob: float
    ) -> tuple[float, float, float]:
        """
        Calculate bookmaker odds from true probabilities with vig margin.
        Returns (home_odds, draw_odds, away_odds).
        """
        # Add vig margin
        total_prob = home_prob + draw_prob + away_prob
        home_prob_adj = home_prob / total_prob * (1 + self.vig_margin)
        draw_prob_adj = draw_prob / total_prob * (1 + self.vig_margin)
        away_prob_adj = away_prob / total_prob * (1 + self.vig_margin)

        # Convert to decimal odds
        home_odds = 1 / home_prob_adj
        draw_odds = 1 / draw_prob_adj
        away_odds = 1 / away_prob_adj

        return round(home_odds, 2), round(draw_odds, 2), round(away_odds, 2)

    def _estimate_true_probabilities(
        self, home_goals: int, away_goals: int
    ) -> tuple[float, float, float]:
        """
        Estimate true match probabilities based on expected goals.
        Uses simplified Elo-like model.
        """
        # Home advantage factor
        home_strength = home_goals * 1.1
        away_strength = away_goals

        # Simple probability estimation
        total_strength = home_strength + away_strength + 1.0  # +1 for draw possibility
        home_prob = home_strength / total_strength * 0.85  # 85% for win/loss, 15% for draw
        away_prob = away_strength / total_strength * 0.85
        draw_prob = 0.15

        return home_prob, draw_prob, away_prob

    def generate_matches(self, n_matches: int = 1500) -> pd.DataFrame:
        """
        Generate synthetic match data.
        Returns DataFrame with realistic football statistics.
        """
        logger.info(f"Generating {n_matches} synthetic matches")

        matches = []
        start_date = datetime(2021, 8, 1)  # Start of 2021-2022 season

        for i in range(n_matches):
            # Select teams
            home_team = random.choice(self.teams)
            away_team = random.choice([t for t in self.teams if t != home_team])

            # Generate date (spread across 3 seasons)
            match_date = start_date + timedelta(days=i * 0.6)

            # Generate goals (Poisson distribution)
            home_goals = int(np.random.poisson(self.home_goals_lambda))
            away_goals = int(np.random.poisson(self.away_goals_lambda))

            # Generate shots (normal distribution)
            home_shots = max(0, int(np.random.normal(self.home_shots_mean, self.home_shots_std)))
            away_shots = max(0, int(np.random.normal(self.away_shots_mean, self.away_shots_std)))

            # Calculate odds based on expected performance
            true_home_prob, true_draw_prob, true_away_prob = self._estimate_true_probabilities(
                self.home_goals_lambda, self.away_goals_lambda
            )

            # Opening odds (B365)
            b365h, b365d, b365a = self._calculate_odds(
                true_home_prob, true_draw_prob, true_away_prob
            )

            # Closing odds (BS) - slight movement
            movement = random.uniform(-0.05, 0.05)
            bsh = round(b365h * (1 + movement), 2)
            bsd = round(b365d * (1 + movement), 2)
            bsa = round(b365a * (1 + movement), 2)

            matches.append(
                {
                    "date": match_date.strftime("%Y-%m-%d"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_shots": home_shots,
                    "away_shots": away_shots,
                    "B365H": b365h,
                    "B365D": b365d,
                    "B365A": b365a,
                    "BSH": bsh,
                    "BSD": bsd,
                    "BSA": bsa,
                }
            )

        df = pd.DataFrame(matches)
        logger.info(f"Generated {len(df)} matches")
        return df

    def save_to_csv(self, df: pd.DataFrame, filename: str = "mock_dataset.csv") -> Path:
        """Save DataFrame to CSV file."""
        output_dir = DATA_RAW_DIR / "historical"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved mock dataset to {output_path}")
        return output_path


def main():
    """Generate and save mock dataset."""
    generator = MockDataGenerator()
    df = generator.generate_matches(1500)
    output_path = generator.save_to_csv(df)
    print(f"Mock dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
