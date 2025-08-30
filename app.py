import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import time
import os
import pickle
from typing import List

# -----------------------------
# Preloaded S&P500 tickers
# -----------------------------
SP500_TICKERS = [
    "AAPL","MSFT","GOOG","AMZN","FB","TSLA","NVDA","BRK-B","JNJ","JPM",
    "V","PG","UNH","HD","DIS","MA","PYPL","BAC","CMCSA","XOM","VZ",
    # ... continue full S&P500 list or a significant subset
]

# -----------------------------
# Technical Indicators
# -----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# -----------------------------
# Fetch stock data
# -----------------------------
def fetch_data(ticker: str, period: str = "240d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    return df

# -----------------------------
# Screen conditions per style
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict):
    df = df.copy()
    df["SMA20"] = sma(df["Close"], params.get("sma20", 20))
    df["SMA50"] = sma(df["Close"], params.get("sma50", 50))
    df["SMA200"] = sma(df["Close"], params.get("sma200", 200))
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL20"] = sma(df["Volume"], 20)

    if df.shape[0] < max(50, params.get("lookback_days", 30)+1):
        return False, {}

    latest = df.iloc[-1]
    partial = {}

    if style == "Momentum + Breakout":
        cond_ma = latest["SMA20"] > latest["SMA50"]
        cond_rsi = 40 <= latest["RSI14"] <= 60
        cond_vol = latest["Volume"] > latest["VOL20"]
        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        cond_breakout = (latest["Close"] > consolidation_high) and (latest["Close"] <= consolidation_high*(1+params["breakout_buffer"]))
        partial = {
            "SMA20>SMA50": cond_ma,
            "RSI40-60": cond_rsi,
            "Volume>VOL20": cond_vol,
            "Breakout": cond_breakout
        }
        return cond_ma and cond_rsi and cond_vol and cond_breakout, partial

    elif style == "Pullback":
        cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"]
        cond_rsi = latest["RSI14"] < 50
        partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
        return cond_pullback and cond_rsi, partial

    elif style == "MA Crossover":
        cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"]
        partial = {"MA Crossover": cond_ma}
        return cond_ma, partial

    elif style == "RSI Range":
        cond_rsi = 40 <= latest["RSI14"] <= 60
        partial = {"RSI40-60": cond_rsi}
        return cond_rsi, partial

    return False, partial

# -----------------------------
# Caching mechanism
# -----------------------------
CACHE_FILE = "sp500_data.pkl"
TIMESTAMP_FILE = "last_fetch_time.txt"

def save_cache(data_dict):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data_dict, f)
    with open(TIMESTAMP_FILE, "w") as f:
        f.write(str(time.time()))

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def cache_expired(days: int = 7) -> bool:
    if os.path.exists(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, "r") as f:
            last_fetch = float(f.read())
        return (time.time() - last_fetch) > days*24*3600
    return True

# -----------------------------
# Screen stocks
# -----------------------------
def screen_stocks(tickers: List[str], style: str, params: dict) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)

    # Load cached data if available and fresh
    if not cache_expired():
        st.info("Loading cached data...")
        data_dict = load_cache()
    else:
        st.info("Fetching latest data for tickers...")
        data_dict = {}
        for i, t in enumerate(tickers):
            df = fetch_data(t)
            if df is not None:
                data_dict[t] = df
            progress.progress((i+1)/total)
        save_cache(data_dict)

    # Scan each stock
    for i, t in enumerate(tickers):
        df = data_dict.get(t)
        if df is None or df.empty:
            continue
        try:
            meets, partial = check_conditions(df, style, params)
            latest = df.iloc[-1]
            results.append({
                "Ticker": t,
                "Meets_Entry": meets,
                "Close": round(latest["Close"],2),
                "RSI14": round(rsi(df["Close"]).iloc[-1],2),
                "SMA20": round(sma(df["Close"],20).iloc[-1],2),
                "SMA50": round(sma(df["Close"],50).iloc[-1],2),
                "Volume": int(latest["Volume"]),
                "VOL20": int(sma(df["Volume"],20).iloc[-1]),
                "Partial_Match": partial
            })
        except Exception as e:
            st.warning(f"Skipping {t}: {e}")
        progress.progress((i+1)/total)

    return pd.DataFrame(results)

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("Multi-Style Stock Screener (S&P500)")

style = st.sidebar.selectbox("Select Trading Style:", ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"])

params = {}
params["lookback_days"] = st.sidebar.slider("Lookback Days (Momentum/Breakout)", 10, 60, 30)
params["breakout_buffer"] = st.sidebar.slider("Breakout Buffer (%)", 0.01, 0.10, 0.05)
params["sma20"] = st.sidebar.slider("SMA20", 10, 50, 20)
params["sma50"] = st.sidebar.slider("SMA50", 20, 100, 50)
params["sma200"] = st.sidebar.slider("SMA200", 50, 250, 200)

if st.sidebar.button("Run Screener"):
    df_results = screen_stocks(SP500_TICKERS, style, params)
    if df_results.empty:
        st.warning("No valid stock data returned. Try again later or adjust filters.")
    else:
        matches = df_results[df_results["Meets_Entry"]]
        st.success(f"Found {len(matches)} matches!")
        st.dataframe(matches)

        if not matches.empty:
            csv = matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matches as CSV",
                data=csv,
                file_name=f"{style.replace(' ','_')}_matches.csv",
                mime="text/csv"
            )



