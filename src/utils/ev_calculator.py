"""
Expected Value (EV+) Calculator
ValueBettingEngine: compares model probabilities vs bookmaker odds.
Calculates EV and optimal stake using Fractional Kelly Criterion.
Filters: EV > 3% and odds > 1.50.
"""

from dataclasses import dataclass
from typing import Any

from src.utils.logger import logger

MIN_EV: float = 0.03
MIN_ODDS: float = 1.50
KELLY_FRACTION: int = 4


@dataclass
class ValueBet:
    """Represents a value betting opportunity."""

    match: str
    market: str
    selection: str
    model_prob: float
    implied_prob: float
    odds_decimal: float
    expected_value: float
    kelly_stake_pct: float
    edge: float

    def __str__(self) -> str:
        return (
            f"[{self.market}] {self.match} -> {self.selection} | "
            f"Odds: {self.odds_decimal:.2f} | "
            f"Model: {self.model_prob:.1%} vs Implied: {self.implied_prob:.1%} | "
            f"EV: {self.expected_value:+.1%} | "
            f"Kelly: {self.kelly_stake_pct:.2%}"
        )


class ValueBettingEngine:
    """
    Calculates Expected Value and Kelly stake for betting opportunities.
    Filters by minimum EV threshold and minimum odds.
    """

    def __init__(
        self,
        min_ev: float = MIN_EV,
        min_odds: float = MIN_ODDS,
        kelly_divisor: int = KELLY_FRACTION,
    ) -> None:
        self.min_ev = min_ev
        self.min_odds = min_odds
        self.kelly_divisor = kelly_divisor

    @staticmethod
    def _odds_to_implied_prob(odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 1.0:
            return 1.0
        return 1.0 / odds

    @staticmethod
    def _calculate_ev(prob: float, odds: float) -> float:
        """Calculate Expected Value: EV = (prob * odds) - 1."""
        return (prob * odds) - 1.0

    @staticmethod
    def _calculate_kelly(prob: float, odds: float, divisor: int = KELLY_FRACTION) -> float:
        """
        Calculate Fractional Kelly stake as percentage of bankroll.
        Kelly = (prob * (odds - 1) - (1 - prob)) / (odds - 1)
        Fractional Kelly = Kelly / divisor
        """
        if odds <= 1.0:
            return 0.0

        b = odds - 1.0
        q = 1.0 - prob
        kelly = (prob * b - q) / b

        fractional = kelly / divisor
        return max(0.0, min(fractional, 1.0))

    def evaluate_1x2(
        self,
        match_name: str,
        model_probs: dict[str, float],
        odds: dict[str, float],
    ) -> list[ValueBet]:
        """
        Evaluate 1X2 market for value bets.
        model_probs: {"H": 0.45, "D": 0.25, "A": 0.30}
        odds: {"H": 2.10, "D": 3.40, "A": 3.50}
        """
        bets: list[ValueBet] = []

        for selection in ["H", "D", "A"]:
            model_prob = model_probs.get(selection, 0.0)
            odd = odds.get(selection, 0.0)

            if odd < self.min_odds:
                continue

            implied = self._odds_to_implied_prob(odd)
            ev = self._calculate_ev(model_prob, odd)

            if ev < self.min_ev:
                continue

            kelly = self._calculate_kelly(model_prob, odd, self.kelly_divisor)
            edge = model_prob - implied

            bet = ValueBet(
                match=match_name,
                market="1X2",
                selection=selection,
                model_prob=model_prob,
                implied_prob=implied,
                odds_decimal=odd,
                expected_value=ev,
                kelly_stake_pct=kelly,
                edge=edge,
            )
            bets.append(bet)

        return bets

    def evaluate_over_under(
        self,
        match_name: str,
        model_prob_over: float,
        odds_over: float,
        odds_under: float,
    ) -> list[ValueBet]:
        """Evaluate Over/Under 2.5 market for value bets."""
        bets: list[ValueBet] = []

        for selection, prob, odd in [
            ("Over 2.5", model_prob_over, odds_over),
            ("Under 2.5", 1.0 - model_prob_over, odds_under),
        ]:
            if odd < self.min_odds:
                continue

            implied = self._odds_to_implied_prob(odd)
            ev = self._calculate_ev(prob, odd)

            if ev < self.min_ev:
                continue

            kelly = self._calculate_kelly(prob, odd, self.kelly_divisor)
            edge = prob - implied

            bet = ValueBet(
                match=match_name,
                market="Over/Under 2.5",
                selection=selection,
                model_prob=prob,
                implied_prob=implied,
                odds_decimal=odd,
                expected_value=ev,
                kelly_stake_pct=kelly,
                edge=edge,
            )
            bets.append(bet)

        return bets

    def scan_all(
        self,
        matches: list[dict[str, Any]],
    ) -> list[ValueBet]:
        """
        Scan all matches for value bets.
        Each match dict should contain:
        - match_name: str
        - model_probs_1x2: dict[str, float]
        - odds_1x2: dict[str, float]
        - model_prob_over: float
        - odds_over: float
        - odds_under: float
        """
        all_bets: list[ValueBet] = []

        for match in matches:
            name = match.get("match_name", "Unknown")

            bets_1x2 = self.evaluate_1x2(
                name,
                match.get("model_probs_1x2", {}),
                match.get("odds_1x2", {}),
            )

            bets_ou = self.evaluate_over_under(
                name,
                match.get("model_prob_over", 0.5),
                match.get("odds_over", 0.0),
                match.get("odds_under", 0.0),
            )

            all_bets.extend(bets_1x2)
            all_bets.extend(bets_ou)

        all_bets.sort(key=lambda b: b.expected_value, reverse=True)
        logger.info(f"Found {len(all_bets)} value bets from {len(matches)} matches")
        return all_bets


def main() -> None:
    """Demo usage."""
    engine = ValueBettingEngine()

    sample_match = {
        "match_name": "Arsenal vs Chelsea",
        "model_probs_1x2": {"H": 0.50, "D": 0.25, "A": 0.25},
        "odds_1x2": {"H": 1.90, "D": 3.60, "A": 4.00},
        "model_prob_over": 0.60,
        "odds_over": 1.85,
        "odds_under": 2.00,
    }

    bets = engine.scan_all([sample_match])

    for bet in bets:
        logger.info(str(bet))


if __name__ == "__main__":
    main()
