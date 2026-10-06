"""
Evaluation and Backtest Module
Calculates regression metrics (RMSE, MAPE), generates publication-quality
Train/Val/Predictions charts (both static Matplotlib and interactive Plotly),
and forecasts tomorrow's stock price.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict


def evaluate_predictions(y_actual: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics between actual and predicted prices.
    """
    y_actual = y_actual.flatten()
    y_pred = y_pred.flatten()
    
    rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))
    mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-9))) * 100
    
    metrics = {
        "RMSE": float(rmse),
        "MAPE (%)": float(mape)
    }
    return metrics


def plot_predictions(
    df: pd.DataFrame,
    train_len: int,
    predictions: np.ndarray,
    ticker: str = "AAPL",
    save_path: str = None
) -> plt.Figure:
    """
    Plots Train, Val (Actual), and Predictions on the same chart,
    reproducing the exact 3-line visual from the user's notebook.
    
    - Blue line: Historical training data (Train)
    - Red line: Actual market prices in test set (Val)
    - Yellow line: Model forecast (Predictions)
    """
    train = df.iloc[:train_len].copy()
    valid = df.iloc[train_len:].copy()
    valid['Predictions'] = predictions.flatten()
    
    available_styles = plt.style.available
    if 'fivethirtyeight' in available_styles:
        plt.style.use('fivethirtyeight')
    else:
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in available_styles else 'default')
        
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('Model', fontsize=18, fontweight='bold', pad=12)
    ax.set_xlabel('Date', fontsize=16, labelpad=10)
    ax.set_ylabel('Close Price USD ($)', fontsize=16, labelpad=10)
    
    ax.plot(train.index, train['Close'], label='Train', color='#008fd5', lw=2.2)
    ax.plot(valid.index, valid['Close'], label='Val', color='#fc4f30', lw=2.2)
    ax.plot(valid.index, valid['Predictions'], label='Predictions', color='#e5ae38', lw=2.2)
    
    ax.legend(['Train', 'Val', 'Predictions'], loc='lower right', fontsize=13, frameon=True)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Evaluate] Plot saved to {save_path}")
        
    return fig


def plot_predictions_interactive(
    df: pd.DataFrame,
    train_len: int,
    predictions: np.ndarray,
    ticker: str = "AAPL"
) -> go.Figure:
    """
    Interactive Plotly chart allowing users to zoom in, pan, hover,
    and use a range slider to closely inspect predictions vs actuals.
    """
    train = df.iloc[:train_len].copy()
    valid = df.iloc[train_len:].copy()
    valid['Predictions'] = predictions.flatten()
    
    fig = go.Figure()
    
    # 1. Historical Train (Blue)
    fig.add_trace(go.Scatter(
        x=train.index,
        y=train['Close'],
        name='Train',
        mode='lines',
        line=dict(color='#008fd5', width=2),
        hovertemplate='<b>Train</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # 2. Actual Validation (Red/Coral)
    fig.add_trace(go.Scatter(
        x=valid.index,
        y=valid['Close'],
        name='Val (Actual)',
        mode='lines',
        line=dict(color='#fc4f30', width=2.2),
        hovertemplate='<b>Val (Actual)</b>: $%{y:.2f}<extra></extra>'
    ))
    
    # 3. Model Predictions (Yellow/Gold)
    fig.add_trace(go.Scatter(
        x=valid.index,
        y=valid['Predictions'],
        name='Predictions',
        mode='lines',
        line=dict(color='#e5ae38', width=2.2, dash='solid'),
        hovertemplate='<b>Predictions</b>: $%{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Model: {ticker} Price Prediction (Interactive Zoom & Pan)",
            font=dict(size=18)
        ),
        xaxis=dict(
            title="Date",
            rangeslider=dict(visible=True),  # Interactive range slider at bottom
            rangeselector=dict(
                buttons=list([
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="Close Price USD ($)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        height=620,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


def predict_tomorrow(model, last_60_days: np.ndarray) -> float:
    """
    Predicts next day's closing price from the recent 60-day price array.
    """
    last_60 = np.array(last_60_days).flatten()
    ref_price = last_60[-1]
    
    normalized_window = (last_60 / (ref_price + 1e-9)).reshape(1, -1)
    
    pred_ratio = float(model.predict(normalized_window)[0])
    pred_price = ref_price * pred_ratio
    return float(pred_price)
