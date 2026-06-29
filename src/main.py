"""
Sports Betting Bot - Main Entry Point
Flow: Load model -> The Odds API (today) -> Calc features -> EV calculator -> Print summary.
"""

import sys
from typing import Any

import pandas as pd

from src.etl.odds_extractor import OddsExtractor
from src.models.predictor import Predictor
from src.utils.ev_calculator import ValueBettingEngine
from src.utils.logger import logger


def _build_features_for_match(
    odds_row: dict[str, Any],
) -> dict[str, float]:
    """
    Build feature dict for a single match.
    Uses rolling averages from DB or defaults.
    """
    return {
        "home_score": float(odds_row.get("home_score", 1)),
        "away_score": float(odds_row.get("away_score", 1)),
        "goal_diff": float(odds_row.get("goal_diff", 0)),
        "home_goals_for_5": float(odds_row.get("home_goals_for_5", 1.5)),
        "home_goals_against_5": float(odds_row.get("home_goals_against_5", 1.0)),
        "away_goals_for_5": float(odds_row.get("away_goals_for_5", 1.2)),
        "away_goals_against_5": float(odds_row.get("away_goals_against_5", 1.3)),
        "home_goals_for_10": float(odds_row.get("home_goals_for_10", 1.5)),
        "home_goals_against_10": float(odds_row.get("home_goals_against_10", 1.0)),
        "away_goals_for_10": float(odds_row.get("away_goals_for_10", 1.2)),
        "away_goals_against_10": float(odds_row.get("away_goals_against_10", 1.3)),
        "goal_form_diff_5": float(odds_row.get("goal_form_diff_5", 0.3)),
        "goal_form_diff_10": float(odds_row.get("goal_form_diff_10", 0.3)),
        "home_goal_net_5": float(odds_row.get("home_goal_net_5", 0.5)),
        "away_goal_net_5": float(odds_row.get("away_goal_net_5", -0.1)),
        "home_goal_net_10": float(odds_row.get("home_goal_net_10", 0.5)),
        "away_goal_net_10": float(odds_row.get("away_goal_net_10", -0.1)),
        "home_elo": float(odds_row.get("home_elo", 1500.0)),
        "away_elo": float(odds_row.get("away_elo", 1500.0)),
        "elo_diff": float(odds_row.get("home_elo", 1500.0))
        - float(odds_row.get("away_elo", 1500.0)),
    }


def _extract_odds_for_event(odds_df: pd.DataFrame, event_id: str) -> dict[str, Any]:
    """Extract 1X2 and Over/Under odds for a single event."""
    event_odds = odds_df[odds_df["event_id"] == event_id]
    if event_odds.empty:
        return {}

    home_team = event_odds.iloc[0]["home_team"]
    away_team = event_odds.iloc[0]["away_team"]
    match_name = f"{home_team} vs {away_team}"

    odds_1x2: dict[str, float] = {}
    odds_totals: dict[str, float] = {}

    h2h_odds = event_odds[event_odds["market"] == "h2h"]
    for _, row in h2h_odds.iterrows():
        sel = str(row["selection"])
        odd = float(row["odds_decimal"])
        if sel == home_team:
            odds_1x2["H"] = odd
        elif sel == away_team:
            odds_1x2["A"] = odd
        elif sel == "Draw":
            odds_1x2["D"] = odd

    totals_odds = event_odds[event_odds["market"] == "totals"]
    for _, row in totals_odds.iterrows():
        sel = str(row["selection"])
        odd = float(row["odds_decimal"])
        if sel == "Over":
            odds_totals["over"] = odd
        elif sel == "Under":
            odds_totals["under"] = odd

    return {
        "match_name": match_name,
        "odds_1x2": odds_1x2,
        "odds_over": odds_totals.get("over", 0.0),
        "odds_under": odds_totals.get("under", 0.0),
    }


def run() -> None:
    """Main execution flow."""
    logger.info("=" * 60)
    logger.info("SPORTS BETTING BOT - Starting analysis")
    logger.info("=" * 60)

    try:
        predictor = Predictor()
        predictor.load_models()
        logger.info("Models loaded successfully")
    except FileNotFoundError:
        logger.error("Models not found. Run training first: python -m src.models.predictor")
        sys.exit(1)

    try:
        extractor = OddsExtractor()
        odds_df = extractor.extract("soccer_epl")
        if odds_df.empty:
            logger.warning("No odds data available for today")
            sys.exit(0)

        events = extractor.get_unique_events(odds_df)
        logger.info(f"Found {len(events)} upcoming matches")
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        sys.exit(1)

    engine = ValueBettingEngine()
    matches_for_ev: list[dict[str, Any]] = []

    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        odds_data = _extract_odds_for_event(odds_df, event_id)

        if not odds_data or not odds_data.get("odds_1x2"):
            continue

        features = _build_features_for_match(odds_data)

        try:
            probs_1x2 = predictor.predict_1x2(features)
            prob_over = predictor.predict_over_under(features)
        except Exception as e:
            logger.warning(f"Prediction failed for {odds_data['match_name']}: {e}")
            continue

        match_entry: dict[str, Any] = {
            "match_name": odds_data["match_name"],
            "model_probs_1x2": probs_1x2,
            "odds_1x2": odds_data["odds_1x2"],
            "model_prob_over": prob_over,
            "odds_over": odds_data.get("odds_over", 0.0),
            "odds_under": odds_data.get("odds_under", 0.0),
        }
        matches_for_ev.append(match_entry)

    if not matches_for_ev:
        logger.info("No matches with sufficient data for EV analysis")
        sys.exit(0)

    value_bets = engine.scan_all(matches_for_ev)

    logger.info("=" * 60)
    logger.info("VALUE BETTING SUMMARY")
    logger.info("=" * 60)

    if not value_bets:
        logger.info("No value bets found (EV > 3% and odds > 1.50)")
    else:
        for i, bet in enumerate(value_bets, 1):
            logger.info(f"  #{i}: {bet}")

    logger.info("=" * 60)
    logger.info(f"Analyzed {len(matches_for_ev)} matches | Value bets: {len(value_bets)}")
    logger.info("=" * 60)


def main() -> None:
    """Entry point."""
    run()


if __name__ == "__main__":
    main()
