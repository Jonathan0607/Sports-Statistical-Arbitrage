import numpy as np
import pandas as pd
from arch import arch_model
import logging

logger = logging.getLogger("PlayerVolatilityTracker")

class PlayerVolatilityTracker:
    def __init__(self):
        """
        GARCH(1,1) Volatility Tracker for Player Performance.
        Models conditional variance: sigma_t^2 = omega + alpha * epsilon_{t-1}^2 + beta * sigma_{t-1}^2
        """
        self.fitted_models = {}

    def get_current_regime_variance(self, player_id: str, history_array: list) -> float:
        """
        Analyzes a player's recent performance history to determine their current volatility regime.
        If they are in a highly volatile streak (boom-or-bust), this will output a high variance,
        which can be injected into the ZINB distribution to widen the tails.
        """
        if len(history_array) < 10:
            logger.warning(f"Insufficient history for {player_id}. Falling back to standard sample variance.")
            return np.var(history_array) if len(history_array) > 1 else 1.0

        # Calculate percentage deviations from a simple rolling mean to represent "returns/residuals"
        series = pd.Series(history_array)
        rolling_mean = series.ewm(span=5, min_periods=1).mean()
        
        # Avoid division by zero
        residuals = series - rolling_mean
        
        # Rescale the series by 100 to help the GARCH optimizer converge
        scale = 100.0 if residuals.std() < 1 else 1.0
        scaled_residuals = residuals * scale
        
        try:
            model = arch_model(scaled_residuals, vol='Garch', p=1, q=1, rescale=False)
            res = model.fit(disp='off')
            
            # Forecast 1-step ahead variance
            forecast = res.forecast(horizon=1)
            forecasted_var = forecast.variance.iloc[-1, 0] / (scale ** 2)
            
            # Cache the model
            self.fitted_models[player_id] = res
            
            logger.info(f"[{player_id}] Volatility Regime Variance: {forecasted_var:.4f}")
            return float(forecasted_var)
            
        except Exception as e:
            logger.error(f"GARCH fitting failed for {player_id}: {e}. Returning sample variance.")
            return float(np.var(history_array))
