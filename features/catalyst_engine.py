import logging
import numpy as np

logger = logging.getLogger("CatalystEngine")

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
            # Pro-rata distribution of the missing usage
            usage_share = base_usage / total_active_usage
            adjusted_usages[player] = base_usage + (missing_player_usage * usage_share)
            
        logger.info(f"Redistributed {missing_player_usage}% usage across {len(active_roster_usages)} players.")
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
            
        # Calculate shrinkage factor (weight given to the player's actual data)
        # Higher sample size -> weight approaches 1.
        # Higher player variance -> weight shrinks toward 0.
        weight = prior_variance / (prior_variance + (player_variance / sample_size))
        
        shrunk_mean = (weight * player_mean) + ((1 - weight) * prior_mean)
        return shrunk_mean

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test Usage Recalculation (Luka is OUT, 35% usage needs redistribution)
    recalc = UsageRecalculator()
    active_team = {"Kyrie Irving": 28.0, "PJ Washington": 15.0, "Dereck Lively": 12.0}
    new_usage = recalc.redistribute_usage(35.0, active_team)
    print(f"Adjusted Usages: {new_usage}")
    
    # Test Bayesian Shrinkage (Backup PG with only 3 games played)
    bayes = BayesianShrinkage()
    shrunk = bayes.shrink_to_prior(player_mean=22.0, player_variance=50.0, prior_mean=10.0, prior_variance=5.0, sample_size=3)
    print(f"Bayesian Shrunk Projection: {shrunk}")
