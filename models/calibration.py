import logging
import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger("ProbabilityCalibrator")

class ProbabilityCalibrator:
    def __init__(self):
        """
        Uses Isotonic Regression to map raw, mathematically derived probabilities 
        to perfectly calibrated empirical frequencies.
        """
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, true_outcomes: np.ndarray):
        """
        Fits a piecewise constant non-decreasing function to smooth prediction bias.
        
        raw_probs: 1D array of predicted probabilities (e.g., from the ZINB model)
        true_outcomes: 1D array of binary outcomes (1 for OVER hit, 0 for miss)
        """
        try:
            self.ir.fit(raw_probs, true_outcomes)
            self.is_fitted = True
            logger.info("Isotonic Regression curve fitted successfully.")
        except Exception as e:
            logger.error(f"Failed to fit Probability Calibrator: {e}")

    def calibrate(self, raw_prob: float) -> float:
        """
        Returns the calibrated probability.
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted. Returning uncalibrated probability.")
            return float(raw_prob)
            
        calibrated = self.ir.predict([raw_prob])[0]
        return float(calibrated)
