"""
Model Architectures Module
Provides regression models to predict future stock prices based on historical windows.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb


class StockPredictor:
    """
    Standard interface for training models and making stock price predictions.
    """
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        self.model = self._build_model()
        
    def _build_model(self):
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        elif self.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == "linear":
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: '{self.model_type}'. Supported: 'xgboost', 'random_forest', 'linear'")

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """Trains the model on sequence features."""
        print(f"[Model] Training {self.model_type.upper()} model on {len(x_train)} samples...")
        self.model.fit(x_train, y_train)
        print("[Model] Training complete.")

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Generates predictions on test sequences."""
        return self.model.predict(x_test)


def get_model(model_type: str = "xgboost") -> StockPredictor:
    return StockPredictor(model_type=model_type)
