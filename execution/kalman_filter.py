import logging
import numpy as np

logger = logging.getLogger("KalmanTracker")

class SharpMoneyTracker:
    def __init__(self, initial_prob: float, process_variance=1e-5, measurement_variance=1e-3):
        # State estimate (the "true" probability)
        self.x = initial_prob
        # Estimate uncertainty (covariance)
        self.P = 1.0 
        # Process noise (how fast we expect the true state to change)
        self.Q = process_variance
        # Measurement noise (how much noise is in the bookmaker's feed)
        self.R = measurement_variance

    def update(self, measurement: float) -> float:
        """Standard 1D Kalman Filter Predict/Update loop."""
        # 1. Predict (State transition is 1 for a random walk)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 2. Update
        # Kalman Gain: How much we trust the measurement vs our prediction
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        logger.debug(f"Filtered Prob: {self.x:.4f} (Measurement: {measurement:.4f}, Gain: {K:.4f})")
        return self.x

    def detect_stale_line(self, retail_prob: float, threshold: float = 0.02) -> bool:
        """Flags if the retail line is significantly lagging the filtered sharp trajectory."""
        discrepancy = abs(self.x - retail_prob)
        if discrepancy > threshold:
            logger.info(f"🚨 STALE LINE DETECTED: Filtered Sharp ({self.x:.4f}) vs Retail ({retail_prob:.4f})")
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Simulate a line opening at 50%
    tracker = SharpMoneyTracker(initial_prob=0.50)
    
    # Sharp money steadily pushes the line, but with noise
    noisy_feed = [0.51, 0.50, 0.52, 0.51, 0.54, 0.53, 0.55]
    print("Processing Sharp Book Feed...")
    for tick in noisy_feed:
        tracker.update(tick)
        
    # Check if a retail book sitting at 50% is now stale
    tracker.detect_stale_line(retail_prob=0.50)
