import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("StatisticalBaselines")

class BaselineFeatureProcessor:
    @staticmethod
    def calculate_ewma(data_series: pd.Series, span: int = 5) -> pd.Series:
        """Calculates Exponentially Weighted Moving Average for recent form."""
        return data_series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def project_game_pace(team_pace: float, opp_pace: float, league_avg_pace: float) -> float:
        """
        Projects total game possessions using the Pythagorean pace expectation.
        Formula: (Team Pace * Opponent Pace) / League Average Pace
        """
        if league_avg_pace == 0:
            return 0
        return (team_pace * opp_pace) / league_avg_pace

    @staticmethod
    def adjust_for_defense(player_baseline: float, opp_def_eff: float, league_avg_def: float) -> float:
        """
        Scales the player's baseline projection against the opponent's 
        defensive efficiency specific to the player's position.
        """
        if league_avg_def == 0:
            return player_baseline
        # If Opp Def Eff is lower (better), the multiplier shrinks the baseline
        adjustment_factor = opp_def_eff / league_avg_def
        return player_baseline * adjustment_factor

if __name__ == "__main__":
    # Quick test
    processor = BaselineFeatureProcessor()
    projected_pace = processor.project_game_pace(102.5, 98.0, 100.0)
    print(f"Projected Game Pace: {projected_pace}")
