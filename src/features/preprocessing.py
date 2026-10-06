"""
Feature Engineering & Preprocessing Module
Extracts closing prices, splits train/test sets, and creates rolling
lookback sequences with window-relative normalization to allow models
to extrapolate to new all-time highs without flatlining.
"""
import math
import numpy as np
import pandas as pd
from typing import Tuple


def prepare_stock_sequences(
    df: pd.DataFrame,
    lookback_days: int = 60,
    train_split: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Transforms stock Close prices into train/test sequence arrays.
    
    Uses window-relative normalization:
    Each 60-day window is normalized by the last price in that window (day t),
    and the target is the ratio of day t+1 to day t.
    This guarantees that models (Random Forest, XGBoost, Linear) can predict
    all-time highs and future market rallies without hitting tree-saturation flatlines.
    
    Parameters:
        df (pd.DataFrame): Dataframe containing 'Close' column.
        lookback_days (int): Number of past days in the window (default 60).
        train_split (float): Train partition ratio (default 0.8).
        
    Returns:
        x_train (np.ndarray): Shape (n_train, lookback_days)
        y_train (np.ndarray): Target ratios for next day
        x_test  (np.ndarray): Shape (n_test, lookback_days)
        y_test  (np.ndarray): Actual unscaled test dollar prices
        test_base_prices (np.ndarray): Price at day t to scale ratio back to dollar price
        train_len (int): Cutoff index for training data
    """
    close_prices = df['Close'].values
    train_len = math.ceil(len(close_prices) * train_split)
    
    X = []
    y_ratios = []
    base_prices = []
    
    for i in range(lookback_days, len(close_prices)):
        window = close_prices[i - lookback_days:i]
        ref_price = window[-1] # The price on day t
        
        # Normalize window relative to day t (e.g. values around 0.85 to 1.05)
        X.append(window / (ref_price + 1e-9))
        # Target ratio: Close[t+1] / Close[t]
        y_ratios.append(close_prices[i] / (ref_price + 1e-9))
        base_prices.append(ref_price)
        
    X = np.array(X)
    y_ratios = np.array(y_ratios)
    base_prices = np.array(base_prices)
    
    split_idx = train_len - lookback_days
    
    x_train = X[:split_idx]
    y_train = y_ratios[:split_idx]
    
    x_test = X[split_idx:]
    test_base_prices = base_prices[split_idx:]
    y_test = close_prices[train_len:] # Actual dollar closing prices in test set
    
    print(f"[Preprocessor] Training samples: {x_train.shape[0]} | Testing samples: {x_test.shape[0]}")
    return x_train, y_train, x_test, y_test, test_base_prices, train_len
