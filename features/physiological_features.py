import numpy as np
import logging

logger = logging.getLogger("PhysiologicalFeatures")

# Arena Microstructure Metadata Matrix
ARENA_METADATA = {
    "DEN": {"lat": 39.7486, "lon": -105.0075, "altitude": 5280},  # Mile High
    "UTA": {"lat": 40.7683, "lon": -111.9011, "altitude": 4226},  # High Altitude
    "DAL": {"lat": 32.7905, "lon": -96.8103, "altitude": 430},
    "LAL": {"lat": 34.0430, "lon": -118.2673, "altitude": 292},
    "MIA": {"lat": 25.7814, "lon": -80.1870, "altitude": 6}
}

class PhysiologicalFeatureProcessor:
    def __init__(self):
        self.metadata = ARENA_METADATA

    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Optimized haversine formula for spatial travel calculations."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2.0 * np.arcsin(np.sqrt(a))
        miles = 3956.0 * c  # Radius of Earth in miles
        return float(miles)

    def compute_travel_fatigue(self, team_itinerary: list) -> float:
        """Calculates cumulative miles traveled over a sequence of team cities."""
        total_miles = 0.0
        if len(team_itinerary) < 2:
            return total_miles
            
        for i in range(len(team_itinerary) - 1):
            city_a = team_itinerary[i]
            city_b = team_itinerary[i+1]
            
            if city_a in self.metadata and city_b in self.metadata:
                loc_a = self.metadata[city_a]
                loc_b = self.metadata[city_b]
                total_miles += self.calculate_haversine_distance(
                    loc_a["lat"], loc_a["lon"], loc_b["lat"], loc_b["lon"]
                )
        return total_miles

    def generate_fatigue_matrix(self, player_rest_days: int, opponent_rest_days: int, destination_team: str, team_itinerary: list) -> dict:
        """Generates continuous/categorical fatigue vector inputs for tree models."""
        # 1. Rest Disparity Index (RDI)
        rdi = player_rest_days - opponent_rest_days
        
        # 2. Cumulative Travel Mileage
        total_travel_miles = self.compute_travel_fatigue(team_itinerary)
        
        # 3. Altitude Shock Identifier
        altitude_shock = 0
        dest_metadata = self.metadata.get(destination_team, {"altitude": 0})
        
        # True systemic shock occurs if jumping into high alt with minimal rest
        if dest_metadata["altitude"] >= 4000 and player_rest_days <= 1:
            altitude_shock = 1
            
        return {
            "feature_rest_disparity_index": rdi,
            "feature_cumulative_travel_miles": total_travel_miles,
            "feature_altitude_shock_flag": altitude_shock
        }

if __name__ == "__main__":
    processor = PhysiologicalFeatureProcessor()
    # Test Case: Dallas plays in Denver on a back-to-back after traveling from LA
    sample_path = ["DAL", "LAL", "DEN"]
    matrix = processor.generate_fatigue_matrix(
        player_rest_days=1, 
        opponent_rest_days=3, 
        destination_team="DEN", 
        team_itinerary=sample_path
    )
    print(f"Generated Physiological Vector: {matrix}")
