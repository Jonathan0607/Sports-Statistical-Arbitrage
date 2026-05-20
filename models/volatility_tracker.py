import numpy as np
import pandas as pd
from arch import arch_model
import logging

logger = logging.getLogger("VolatilityTracker")

class VolatilityTracker:
    def __init__(self):
        self.model = None
        self.result = None
        self.scale = 1.0
        
    def fit(self, returns_series: pd.Series):
        """
        Fits a GARCH(1,1) model to the given series.
        In sports context, this is typically the residual (actual points - expected points),
        or percentage deviations from the EWMA baseline.
        """
        # Rescale the series by 100 to help the optimizer converge (standard practice for ARCH)
        if returns_series.std() < 1:
            self.scale = 100.0 
            
        scaled_series = returns_series * self.scale
        
        try:
            self.model = arch_model(scaled_series, vol='Garch', p=1, q=1, rescale=False)
            self.result = self.model.fit(disp='off')
            logger.info("GARCH(1,1) model fitted successfully.")
        except Exception as e:
            logger.error(f"Failed to fit GARCH model: {e}")
            raise
            
    def forecast_variance(self, horizon: int = 1) -> float:
        """
        Forecasts the expected variance for the next 'horizon' periods.
        """
        if self.result is None:
            raise ValueError("Model must be fitted before forecasting.")
            
        forecast = self.result.forecast(horizon=horizon)
        # Variance of the 1-step ahead forecast, descaled back to original
        forecasted_var = forecast.variance.iloc[-1, 0] / (self.scale ** 2)
        return float(forecasted_var)
