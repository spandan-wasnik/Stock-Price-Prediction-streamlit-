"""
Data Loader Module
Downloads historical stock data from Yahoo Finance using yfinance.
"""
import yfinance as yf
import pandas as pd


def load_stock_data(ticker: str = "AAPL", start_date: str = "2018-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Downloads historical stock prices for a given ticker and date range.
    
    Parameters:
        ticker (str): The stock symbol (e.g., 'AAPL', 'TSLA', 'MSFT').
        start_date (str): Start date string in 'YYYY-MM-DD' format.
        end_date (str): End date string in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: Stock dataframe containing Open, High, Low, Close, Volume.
    """
    print(f"[DataLoader] Fetching data for {ticker} from {start_date} to {end_date}...")
    df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Please verify the symbol and dates.")
        
    # Ensure standard column naming
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"[DataLoader] Successfully loaded {len(df)} trading days.")
    return df
