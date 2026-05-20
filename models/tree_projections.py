import logging
import xgboost as xgb
import numpy as np

logger = logging.getLogger("PlayerProjectionModel")

class PlayerProjectionModel:
    def __init__(self):
        """
        Gradient Boosted Tree model for forecasting baseline player expected metrics
        based on tabular game logs and categorical interactions.
        """
        # We train two separate trees for Minutes (Volume) and Usage (Rate)
        self.xmin_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            objective="reg:squarederror"
        )
        self.xusg_model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            objective="reg:squarederror"
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_min: np.ndarray, y_usg: np.ndarray):
        """
        Trains the XGBoost models.
        X_train: Matrix of features (e.g., historical stats, matchup strength, categorical flags like 'starting_center_out')
        y_min: Target expected minutes
        y_usg: Target expected usage rate
        """
        try:
            logger.info("Training xMin XGBoost sub-model...")
            self.xmin_model.fit(X_train, y_min)
            
            logger.info("Training xUSG XGBoost sub-model...")
            self.xusg_model.fit(X_train, y_usg)
            
            self.is_fitted = True
            logger.info("XGBoost training complete.")
        except Exception as e:
            logger.error(f"Failed to fit XGBoost models: {e}")
            raise

    def predict(self, X_eval: np.ndarray) -> dict:
        """
        Generates point projections for minutes and usage based on the active state matrix.
        """
        if not self.is_fitted:
            logger.warning("Model not fitted. Returning baseline defaults.")
            return {"xMin": 25.0, "xUSG": 0.20}
            
        x_min = self.xmin_model.predict(X_eval)
        x_usg = self.xusg_model.predict(X_eval)
        
        return {
            "xMin": float(x_min[0]),
            "xUSG": float(x_usg[0])
        }
