# ===============================================
# Prediction Service
# Uses trained CatBoost model + Live yfinance data
# ===============================================

import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas_market_calendars as mcal

# ----------------------------------
# CONFIG
# ----------------------------------

MODEL_PATH = "ml_models/catboost_model.pkl"
MAX_FORECAST_DAYS = 14

# Indian market holidays (NSE)
INDIAN_MARKET_HOLIDAYS_2025 = [
    "2025-01-26",  # Republic Day
    "2025-03-14",  # Holi
    "2025-03-31",  # Ramzan Id
    "2025-04-10",  # Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-05-01",  # Maharashtra Day
    "2025-08-15",  # Independence Day
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-24",  # Dasara
    "2025-11-05",  # Diwali
    "2025-11-15",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
]

INDIAN_MARKET_HOLIDAYS_2026 = [
    "2026-01-26",  # Republic Day
    "2026-03-04",  # Holi
    "2026-03-20",  # Ramzan Id
    "2026-03-30",  # Mahavir Jayanti
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-08-15",  # Independence Day
    "2026-09-16",  # Ganesh Chaturthi
    "2026-10-02",  # Gandhi Jayanti
    "2026-10-13",  # Dasara
    "2026-10-26",  # Diwali
    "2026-11-05",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
]

# Combine holidays
MARKET_HOLIDAYS = set(INDIAN_MARKET_HOLIDAYS_2025 + INDIAN_MARKET_HOLIDAYS_2026)


# ----------------------------------
# LOAD MODEL
# ----------------------------------

model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
feature_cols = model_data["features"]
target_cols = model_data["targets"]


# ----------------------------------
# MARKET CALENDAR FUNCTIONS
# ----------------------------------

def is_market_open(date):
    """Check if the market is open on a given date"""
    # Check if it's a weekend (Saturday or Sunday)
    if date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check if it's a holiday
    date_str = date.strftime("%Y-%m-%d")
    if date_str in MARKET_HOLIDAYS:
        return False
    
    return True

def get_next_market_day(start_date, days_ahead):
    """Get the next market day after skipping holidays and weekends"""
    current_date = start_date
    market_days_found = 0
    result_dates = []
    
    while market_days_found < days_ahead:
        current_date += timedelta(days=1)
        if is_market_open(current_date):
            market_days_found += 1
            result_dates.append(current_date)
    
    return result_dates


# ----------------------------------
# TECHNICAL INDICATORS
# ----------------------------------

def add_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA_7"] = close.rolling(window=7, min_periods=1).mean()
    df["SMA_14"] = close.rolling(window=14, min_periods=1).mean()

    df["EMA_7"] = close.ewm(span=7, adjust=False).mean()
    df["EMA_14"] = close.ewm(span=14, adjust=False).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    
    df["Volatility_14"] = close.rolling(window=14).std()
    
    # Bollinger Bands
    df["BB_Middle"] = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

    return df


# ----------------------------------
# CREATE LAG FEATURES
# ----------------------------------

def create_lag_features(df, lag_days=30):
    feature_base = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_7", "SMA_14",
        "EMA_7", "EMA_14",
        "RSI_14", "Volatility_14",
        "BB_Upper", "BB_Lower", "BB_Middle",
        "MACD", "MACD_Signal", "MACD_Histogram"
    ]

    for lag in range(1, lag_days + 1):
        for col in feature_base:
            if col in df.columns:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


# ----------------------------------
# FETCH COMPANY DATA
# ----------------------------------

def fetch_company_data(symbol):
    df = yf.download(symbol, period="6mo", progress=False)

    if df.empty:
        raise ValueError("No stock data found")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Company"] = symbol

    df = df[["Date", "Company", "Open", "High", "Low", "Close", "Volume"]]

    df = add_indicators(df)
    df = create_lag_features(df)

    df.dropna(inplace=True)

    return df


# ----------------------------------
# PREPARE LAST FEATURE ROW
# ----------------------------------

def prepare_latest_features(df):
    latest_row = df.iloc[-1]
    X = latest_row[feature_cols].values.reshape(1, -1)
    return X


# ----------------------------------
# CALCULATE BUY/SELL LEVELS
# ----------------------------------

def calculate_buy_sell_levels(current_price, predicted_close, volatility):
    """
    Calculate comprehensive buy and sell levels based on predicted price and volatility
    """
    price_change = predicted_close - current_price
    change_percent = (price_change / current_price) * 100
    
    # Calculate support and resistance levels
    support_level = min(current_price, predicted_close) * (1 - volatility/100)
    resistance_level = max(current_price, predicted_close) * (1 + volatility/100)
    
    # Fibonacci retracement levels
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    price_range = abs(predicted_close - current_price)
    
    # Buy levels (below current price)
    buy_levels = {
        "conservative": current_price * (1 - volatility/200),
        "aggressive": current_price * (1 - volatility/100),
        "strong_support": support_level,
        "fibonacci_382": current_price - (price_range * 0.382),
        "fibonacci_500": current_price - (price_range * 0.5),
        "fibonacci_618": current_price - (price_range * 0.618)
    }
    
    # Sell levels (above current price)
    sell_levels = {
        "conservative": current_price * (1 + volatility/200),
        "aggressive": current_price * (1 + volatility/100),
        "strong_resistance": resistance_level,
        "fibonacci_382": current_price + (price_range * 0.382),
        "fibonacci_500": current_price + (price_range * 0.5),
        "fibonacci_618": current_price + (price_range * 0.618)
    }
    
    # Stop loss levels
    stop_loss = {
        "for_buy_position": support_level * 0.98,  # 2% below support
        "for_sell_position": resistance_level * 1.02,  # 2% above resistance
        "trailing_stop": current_price * (1 - volatility/50) if price_change > 0 else current_price * (1 + volatility/50)
    }
    
    # Position sizing suggestion based on risk/reward
    risk_reward_ratio = abs(price_change) / (volatility * current_price / 100) if volatility > 0 else 1
    
    if risk_reward_ratio > 3:
        position_size = "Aggressive (75-100% of capital) - Excellent risk/reward"
    elif risk_reward_ratio > 2:
        position_size = "Moderate (50-75% of capital) - Good risk/reward"
    elif risk_reward_ratio > 1:
        position_size = "Conservative (25-50% of capital) - Fair risk/reward"
    else:
        position_size = "Wait & Watch (0-25% of capital) - Poor risk/reward"
    
    return {
        "buy_levels": {k: round(v, 2) for k, v in buy_levels.items()},
        "sell_levels": {k: round(v, 2) for k, v in sell_levels.items()},
        "stop_loss": {k: round(v, 2) for k, v in stop_loss.items()},
        "position_sizing": position_size,
        "expected_change_percent": round(change_percent, 2),
        "volatility": round(volatility, 2),
        "support_level": round(support_level, 2),
        "resistance_level": round(resistance_level, 2),
        "risk_reward_ratio": round(risk_reward_ratio, 2)
    }


# ----------------------------------
# RECURSIVE FORECAST
# ----------------------------------

def predict_future(symbol, days_ahead):
    if days_ahead < 1 or days_ahead > MAX_FORECAST_DAYS:
        raise ValueError("Days must be between 1 and 14")

    df = fetch_company_data(symbol)

    predictions = []
    trading_dates = []

    # Prepare latest feature row once
    X = prepare_latest_features(df)

    base_date = df["Date"].iloc[-1]
    
    # Get actual trading days (skip holidays/weekends)
    trading_dates = get_next_market_day(base_date, days_ahead)
    
    # Get current price and volatility for buy/sell calculations
    current_price = df["Close"].iloc[-1]
    volatility = df["Volatility_14"].iloc[-1]

    for i, trade_date in enumerate(trading_dates):
        pred = model.predict(X)[0]
        
        # Calculate buy/sell levels for this prediction
        trade_levels = calculate_buy_sell_levels(current_price, pred[3], volatility)
        
        predictions.append({
            "date": str(trade_date.date()),
            "open": float(pred[0]),      # Open
            "high": float(pred[1]),      # High
            "low": float(pred[2]),       # Low
            "close": float(pred[3]),      # Close
            "volume": float(pred[4]) if len(pred) > 4 else 0,  # Volume if available
            "buy_sell_levels": trade_levels
        })
        
        # Update current price for next iteration (using predicted close)
        current_price = float(pred[3])

    return predictions


# ----------------------------------
# NEXT DAY PREDICTION
# ----------------------------------

def predict_next_day(symbol):
    return predict_future(symbol, 1)[0]