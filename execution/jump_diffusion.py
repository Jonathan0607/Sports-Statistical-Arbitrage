import numpy as np
import logging

logger = logging.getLogger("StochasticMarketMonitor")

class StochasticMarketMonitor:
    def __init__(self, theta: float = 0.1, sigma: float = 0.02):
        """
        Models the sharp book implied probability stream using an Ornstein-Uhlenbeck (OU) process 
        with jump diffusion.
        
        dX_t = theta * (mu - X_t)dt + sigma * dW_t + J_t * dN_t
        
        theta: Speed of mean reversion (how fast lines correct back to fair value)
        sigma: Standard deviation of the Wiener process (normal market tick noise)
        """
        self.theta = theta
        self.sigma = sigma
        
    def detect_overreaction(self, current_implied_prob: float, ou_mean: float, time_step_dt: float = 1.0) -> bool:
        """
        Analyzes a sudden change in implied probability and flags if a jump J_t has occurred
        that violates the standard OU process variance bounds.
        
        If an overreaction is detected, the engine flags a mean-reverting opportunity to fade the panic.
        
        current_implied_prob: The latest implied probability of the prop (e.g. 0.65)
        ou_mean: The long-term moving average of the probability (the 'mu' in OU)
        time_step_dt: Time delta since last tick.
        """
        # Expected deviation under normal OU drift
        expected_drift = self.theta * (ou_mean - current_implied_prob) * time_step_dt
        
        # Standard deviation of the noise component
        volatility_bound = self.sigma * np.sqrt(time_step_dt)
        
        # Calculate the actual delta from the mean
        actual_delta = current_implied_prob - ou_mean
        
        # A "Jump" (J_t * dN_t) is detected if the actual delta exceeds the expected drift 
        # plus 3 standard deviations of the normal market noise.
        jump_threshold = abs(expected_drift) + (3 * volatility_bound)
        
        if abs(actual_delta) > jump_threshold:
            logger.warning(
                f"[JUMP DETECTED] Line gapped by {actual_delta*100:.2f}%. "
                f"Exceeds OU noise threshold of {jump_threshold*100:.2f}%. "
                "Flagging for mean-reversion fade."
            )
            return True
            
        return False
