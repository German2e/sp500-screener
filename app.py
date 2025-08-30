import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List
import time

# -----------------------------
# Indicators
# -----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# -----------------------------
# Fetch stock data with retry
# -----------------------------
def fetch_data(ticker: str, period: str = "600d", interval: str = "1d") -> pd.DataFrame:
    for attempt in range(3):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not df.empty:
                return df
        except Exception as e:
            time.sleep(1)
    return pd.DataFrame()  # return empty if all attempts fail

# -----------------------------
# Condition checkers
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> bool:
    if df.empty or df.shape[0] < max(50, params.get("lookback_days", 30)+1):
        return False
    df = df.copy()
    df["SMA20"] = sma(df["Close"], params.get("sma20", 20))
    df["SMA50"] = sma(df["Close"], params.get("sma50", 50))
    df["SMA200"] = sma(df["Close"], params.get("sma200", 200))
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL20"] = sma(df["Volume"], 20)
    latest = df.iloc[-1]

    if style == "Momentum + Breakout":
        cond_ma = latest["SMA20"] > latest["SMA50"]
        cond_rsi = 40 <= latest["RSI14"] <= 60
        cond_vol = latest["Volume"] > latest["VOL20"]
        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        cond_breakout = (latest["Close"] > consolidation_high) and (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"]))
        return bool(cond_ma and cond_rsi and cond_vol and cond_breakout)

    elif style == "Pullback":
        cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"]
        cond_rsi = latest["RSI14"] < 50
        return bool(cond_pullback and cond_rsi)

    elif style == "MA Crossover":
        cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"]
        return bool(cond_ma)

    elif style == "RSI Range":
        return 40 <= latest["RSI14"] <= 60

    return False

# -----------------------------
# Main screener
# -----------------------------
def screen_stocks(tickers: List[str], style: str, **params) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)

    for i, t in enumerate(tickers):
        df = fetch_data(t)
        if df.empty:
            continue
        try:
            meets = check_conditions(df, style, params)
            latest_close = df["Close"].iloc[-1]
            latest_rsi = rsi(df["Close"]).iloc[-1]
            sma20_val = sma(df["Close"], 20).iloc[-1]
            sma50_val = sma(df["Close"], 50).iloc[-1]
            vol = df["Volume"].iloc[-1]
            vol20 = sma(df["Volume"], 20).iloc[-1]
            results.append({
                "Ticker": t,
                "Meets_Entry": meets,
                "Close": round(latest_close, 2),
                "RSI14": round(latest_rsi, 2) if not np.isnan(latest_rsi) else None,
                "SMA20": round(sma20_val, 2) if not np.isnan(sma20_val) else None,
                "SMA50": round(sma50_val, 2) if not np.isnan(sma50_val) else None,
                "Volume": int(vol),
                "VOL20": int(vol20) if not np.isnan(vol20) else None
            })
        except Exception:
            continue
        progress.progress((i+1)/total)
    return pd.DataFrame(results)

# -----------------------------
# Streamlit interface
# -----------------------------
st.title("Momentum & Swing Stock Screener")

style = st.sidebar.selectbox(
    "Select Trading Style:",
    ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"]
)

# Style parameters
params = {}
params["lookback_days"] = st.sidebar.slider("Lookback Days (Momentum/Breakout)", 10, 60, 30)
params["breakout_buffer"] = st.sidebar.slider("Breakout Buffer (%)", 0.01, 0.10, 0.05)
params["sma20"] = st.sidebar.slider("SMA20", 10, 50, 20)
params["sma50"] = st.sidebar.slider("SMA50", 20, 100, 50)
params["sma200"] = st.sidebar.slider("SMA200", 50, 250, 200)

# Input tickers
tickers_input = st.text_area("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOG):")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if st.sidebar.button("Run Screener") and tickers:
    st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")
    df_results = screen_stocks(tickers, style, **params)
    if df_results.empty:
        st.warning("No valid stock data returned. Try again later or adjust filters.")
    else:
        matches = df_results[df_results["Meets_Entry"] == True]
        st.success(f"Found {len(matches)} matches!")
        st.dataframe(matches)

        if not matches.empty:
            csv = matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matches as CSV",
                data=csv,
                file_name=f"{style.replace(' ', '_')}_matches.csv",
                mime="text/csv"
            )



