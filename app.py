"""
Streamlit Web Dashboard for Stock Price Prediction
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from src.data.data_loader import load_stock_data
from src.features.preprocessing import prepare_stock_sequences
from src.models.models import get_model
from src.backtest.evaluate import (
    evaluate_predictions,
    plot_predictions,
    plot_predictions_interactive,
    predict_tomorrow
)

st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Stock Price Prediction Model")
st.markdown("Visualizing historical stock training data alongside validation actuals and model predictions")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️Configuration")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper().strip()
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2012-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2026-01-01"))
lookback_days = st.sidebar.slider("Lookback Window (Days)", min_value=10, max_value=120, value=60)
model_choice = st.sidebar.selectbox("Model Architecture", ["random_forest", "xgboost", "linear"], index=0)

run_button = st.sidebar.button(" Train & Predict", type="primary", use_container_width=True)

if run_button:
    with st.spinner(f"Fetching historical data for {ticker}..."):
        try:
            df = load_stock_data(ticker=ticker, start_date=str(start_date), end_date=str(end_date))
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

    # Preprocessing & Model Execution
    with st.spinner(f"Training {model_choice.upper()} model and generating predictions..."):
        x_train, y_train, x_test, y_test, test_base_prices, train_len = prepare_stock_sequences(
            df=df,
            lookback_days=lookback_days,
            train_split=0.8
        )
        
        # Train
        predictor = get_model(model_type=model_choice)
        predictor.train(x_train, y_train)

        # Predict Test set (ratios converted to dollars)
        pred_ratios = predictor.predict(x_test)
        predictions = test_base_prices * pred_ratios
        
        # Evaluate
        metrics = evaluate_predictions(y_test, predictions)

    # 1. Performance Metrics
    st.subheader("Model Evaluation")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Selected Model", model_choice.upper())
    m_col2.metric("RMSE (Error in $)", f"${metrics['RMSE']:.2f}")
    m_col3.metric("MAPE (Avg Error %)", f"{metrics['MAPE (%)']:.2f}%")
    m_col4.metric("Test Period Days", len(y_test))

    # 2. Zoomable & Classic Chart Visualizations
    st.subheader("Model Predictions vs Actuals")
    tab_interactive, tab_classic = st.tabs([
        "Interactive Zoom & Pan (Plotly)",
        "Classic Notebook View (Matplotlib)"
    ])
    
    with tab_interactive:
        st.caption(" **Zoom controls**: Click & drag on the chart to box-zoom, use the bottom slider / buttons (6m, 1y, 3y, All), scroll wheel to zoom, or double-click to reset view.")
        fig_interactive = plot_predictions_interactive(df, train_len, predictions, ticker=ticker)
        st.plotly_chart(fig_interactive, use_container_width=True)

    with tab_classic:
        fig_static = plot_predictions(df, train_len, predictions, ticker=ticker)
        st.pyplot(fig_static)
        plt.close(fig_static)

    # 3. Next-Day Live Forecast
    st.subheader("Next Day Real-Time Forecast")
    with st.spinner("Fetching latest live quote..."):
        try:
            live_df = yf.Ticker(ticker).history(period="120d", auto_adjust=True)
            if len(live_df) >= lookback_days:
                last_window = live_df['Close'].iloc[-lookback_days:].values
                latest_close = float(live_df['Close'].iloc[-1])
                latest_date = live_df.index[-1].strftime('%Y-%m-%d')
                
                tomorrow_price = predict_tomorrow(predictor, last_window)
                delta_price = tomorrow_price - latest_close
                pct_delta = (delta_price / latest_close) * 100
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Latest Market Date", latest_date)
                c2.metric("Latest Close Price", f"${latest_close:.2f}")
                c3.metric(
                    "Forecast for Next Day",
                    f"${tomorrow_price:.2f}",
                    delta=f"{delta_price:+.2f} ({pct_delta:+.2f}%)"
                )
            else:
                st.warning("Not enough live historical data to form a 60-day prediction window.")
        except Exception as ex:
            st.warning(f"Could not fetch live quote: {ex}")

else:
    st.info("Set your parameters in the sidebar and click **'Train and Predict'** to generate the prediction graph.")
