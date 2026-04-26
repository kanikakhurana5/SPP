from fastapi import APIRouter
import yfinance as yf
import pandas as pd

router = APIRouter()

# ---------------------------------
# Companies For Dashboard
# ---------------------------------

COMPANIES = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS",
    "LT.NS",
    "HINDUNILVR.NS",
    "BAJFINANCE.NS"
]


# ---------------------------------
# Live Stock Dashboard
# ---------------------------------

@router.get("/stocks/live")
def get_live_stocks():

    results = []

    for company in COMPANIES:

        try:
            df = yf.download(company, period="1d", progress=False)

            if df.empty:
                continue

            # Fix MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            latest = df.iloc[-1]

            results.append({
                "company": company,
                "open": float(latest["Open"]),
                "high": float(latest["High"]),
                "low": float(latest["Low"]),
                "close": float(latest["Close"]),
                "volume": int(latest["Volume"])
            })

        except Exception:
            continue

    return results


# ---------------------------------
# Last 14 Days Company Data
# ---------------------------------

@router.get("/stocks/company/{symbol}")
def get_company_data(symbol: str):

    try:
        df = yf.download(symbol.upper(), period="1mo", progress=False)

        if df.empty:
            return []

        # Fix MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)

        # Keep only last 14 days
        df = df.tail(14)

        # Convert to JSON-safe format
        records = []

        for _, row in df.iterrows():
            records.append({
                "Date": str(row["Date"]),
                "Open": float(row["Open"]),
                "High": float(row["High"]),
                "Low": float(row["Low"]),
                "Close": float(row["Close"]),
                "Volume": int(row["Volume"])
            })

        return records

    except Exception as e:
        return {"error": str(e)}
