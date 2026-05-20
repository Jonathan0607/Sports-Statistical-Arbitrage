import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import numpy as np
load_dotenv()
logger = logging.getLogger('FastNewsParser')

class FastNewsParser:

    def __init__(self):
        api_key = os.getenv('LLM_API_KEY')
        if not api_key:
            logger.warning('LLM_API_KEY not found in environment. Parser will fail.')
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = "You are a high-frequency sports data parser. Extract the NBA player's name and their playing status from the following tweet. Respond ONLY in valid JSON format with the keys 'player_name' and 'status'. Standardize 'status' to one of: 'IN', 'OUT', 'QUESTIONABLE', 'PROBABLE', or 'UNKNOWN'."

    def parse_tweet(self, tweet_text: str) -> dict:
        """Parses unstructured text into a deterministic JSON status."""
        try:
            response = self.client.chat.completions.create(model='gpt-4o-mini', messages=[{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': tweet_text}], response_format={'type': 'json_object'}, temperature=0.0)
            parsed_json = json.loads(response.choices[0].message.content)
            logger.info(f"Successfully parsed tweet for {parsed_json.get('player_name')}")
            return parsed_json
        except Exception as e:
            logger.error(f'Failed to parse tweet: {e}')
            return {'player_name': 'unknown', 'status': 'UNKNOWN'}
logger = logging.getLogger('StatisticalBaselines')

class BaselineFeatureProcessor:

    @staticmethod
    def calculate_ewma(data_series: pd.Series, span: int=5) -> pd.Series:
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
        return team_pace * opp_pace / league_avg_pace

    @staticmethod
    def adjust_for_defense(player_baseline: float, opp_def_eff: float, league_avg_def: float) -> float:
        """
        Scales the player's baseline projection against the opponent's 
        defensive efficiency specific to the player's position.
        """
        if league_avg_def == 0:
            return player_baseline
        adjustment_factor = opp_def_eff / league_avg_def
        return player_baseline * adjustment_factor
logger = logging.getLogger('CatalystEngine')

class UsageRecalculator:

    @staticmethod
    def redistribute_usage(missing_player_usage: float, active_roster_usages: dict) -> dict:
        """
        Redistributes the usage rate of a scratched player (e.g., OUT via NLP parser)
        proportionally among the remaining active players on the floor.
        """
        total_active_usage = sum(active_roster_usages.values())
        if total_active_usage == 0:
            return active_roster_usages
        adjusted_usages = {}
        for player, base_usage in active_roster_usages.items():
            usage_share = base_usage / total_active_usage
            adjusted_usages[player] = base_usage + missing_player_usage * usage_share
        logger.info(f'Redistributed {missing_player_usage}% usage across {len(active_roster_usages)} players.')
        return adjusted_usages

class BayesianShrinkage:

    @staticmethod
    def shrink_to_prior(player_mean: float, player_variance: float, prior_mean: float, prior_variance: float, sample_size: int) -> float:
        """
        Hierarchical Bayesian Shrinkage.
        Pulls the highly volatile mean of a low-sample bench player toward the 
        stable positional prior (e.g., average backup PG production).
        """
        if sample_size == 0 or prior_variance == 0:
            return prior_mean
        weight = prior_variance / (prior_variance + player_variance / sample_size)
        shrunk_mean = weight * player_mean + (1 - weight) * prior_mean
        return shrunk_mean
logger = logging.getLogger('PhysiologicalFeatures')
ARENA_METADATA = {'DEN': {'lat': 39.7486, 'lon': -105.0075, 'altitude': 5280}, 'UTA': {'lat': 40.7683, 'lon': -111.9011, 'altitude': 4226}, 'DAL': {'lat': 32.7905, 'lon': -96.8103, 'altitude': 430}, 'LAL': {'lat': 34.043, 'lon': -118.2673, 'altitude': 292}, 'MIA': {'lat': 25.7814, 'lon': -80.187, 'altitude': 6}}

class PhysiologicalFeatureProcessor:

    def __init__(self):
        self.metadata = ARENA_METADATA

    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Optimized haversine formula for spatial travel calculations."""
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        miles = 3956.0 * c
        return float(miles)

    def compute_travel_fatigue(self, team_itinerary: list) -> float:
        """Calculates cumulative miles traveled over a sequence of team cities."""
        total_miles = 0.0
        if len(team_itinerary) < 2:
            return total_miles
        for i in range(len(team_itinerary) - 1):
            city_a = team_itinerary[i]
            city_b = team_itinerary[i + 1]
            if city_a in self.metadata and city_b in self.metadata:
                loc_a = self.metadata[city_a]
                loc_b = self.metadata[city_b]
                total_miles += self.calculate_haversine_distance(loc_a['lat'], loc_a['lon'], loc_b['lat'], loc_b['lon'])
        return total_miles

    def generate_fatigue_matrix(self, player_rest_days: int, opponent_rest_days: int, destination_team: str, team_itinerary: list) -> dict:
        """Generates continuous/categorical fatigue vector inputs for tree models."""
        rdi = player_rest_days - opponent_rest_days
        total_travel_miles = self.compute_travel_fatigue(team_itinerary)
        altitude_shock = 0
        dest_metadata = self.metadata.get(destination_team, {'altitude': 0})
        if dest_metadata['altitude'] >= 4000 and player_rest_days <= 1:
            altitude_shock = 1
        return {'feature_rest_disparity_index': rdi, 'feature_cumulative_travel_miles': total_travel_miles, 'feature_altitude_shock_flag': altitude_shock}