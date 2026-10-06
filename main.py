"""
Main Execution Script
Runs the end-to-end stock prediction pipeline using settings from config.yaml.
"""
import os
import yaml
import yfinance as yf
import matplotlib.pyplot as plt

from src.data.data_loader import load_stock_data
from src.features.preprocessing import prepare_stock_sequences
from src.models.models import get_model
from src.backtest.evaluate import evaluate_predictions, plot_predictions, predict_tomorrow


def main():
    # 1. Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    ticker = config.get("ticker", "AAPL")
    start_date = config.get("start_date", "2014-01-01")
    end_date = config.get("end_date", "2026-01-01")
    lookback_days = config.get("lookback_days", 60)
    train_split = config.get("train_split", 0.8)
    model_type = config.get("model_type", "random_forest")
    
    print("=" * 60)
    print(f"STOCK PRICE PREDICTION PIPELINE: {ticker}")
    print("=" * 60)
    
    # 2. Ingest Data
    df = load_stock_data(ticker=ticker, start_date=start_date, end_date=end_date)
    
    # 3. Preprocess and create 60-day sequences (window-relative normalization)
    x_train, y_train, x_test, y_test, test_base_prices, train_len = prepare_stock_sequences(
        df=df,
        lookback_days=lookback_days,
        train_split=train_split
    )
    
    # 4. Train Model
    predictor = get_model(model_type=model_type)
    predictor.train(x_train, y_train)
    
    # 5. Predict on Test Set
    pred_ratios = predictor.predict(x_test)
    predictions = test_base_prices * pred_ratios
    
    # 6. Evaluate
    metrics = evaluate_predictions(y_test, predictions)
    print("\n--- EVALUATION RESULTS ---")
    for metric_name, val in metrics.items():
        print(f"{metric_name}: {val:.4f}")
        
    # 7. Predict Tomorrow's Price
    print("\n--- NEXT DAY PREDICTION ---")
    live_df = yf.Ticker(ticker).history(period="120d", auto_adjust=True)
    last_60 = live_df['Close'].iloc[-lookback_days:].values
    latest_close = float(live_df['Close'].iloc[-1])
    latest_date = live_df.index[-1].strftime('%Y-%m-%d')
    
    pred_tomorrow = predict_tomorrow(predictor, last_60)
    change = pred_tomorrow - latest_close
    pct_change = (change / latest_close) * 100
    
    print(f"Latest Market Date : {latest_date}")
    print(f"Latest Close Price : ${latest_close:.2f}")
    print(f"Predicted Price    : ${pred_tomorrow:.2f} ({pct_change:+.2f}%)")
    
    # 8. Save Plot matching the notebook visual
    save_fig_path = os.path.join(os.path.dirname(__file__), "prediction_results.png")
    fig = plot_predictions(df, train_len, predictions, ticker=ticker, save_path=save_fig_path)
    plt.close(fig)
    print(f"\nPlot saved to: {save_fig_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
