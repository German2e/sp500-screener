import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
from typing import List

# -----------------------------
# Scrape S&P500 tickers from Finviz
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> list[str]:
    url = "https://finviz.com/screener.ashx?v=111&f=idx_sp500"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")

    tickers = []
    for a in soup.select("a.screener-link-primary"):
        tickers.append(a.text.strip())
    return tickers

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
# Fetch data safely
# -----------------------------
def fetch_data(ticker: str, period: str = "240d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

# -----------------------------
# Condition checkers
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

    # Momentum + Breakout
    if style == "Momentum + Breakout":
        cond_ma = latest["SMA20"] > latest["SMA50"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) else False
        cond_rsi = 40 <= latest["RSI14"] <= 60 if not pd.isna(latest["RSI14"]) else False
        cond_vol = latest["Volume"] > latest["VOL20"] if not pd.isna(latest["VOL20"]) else False
        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        cond_breakout = (latest["Close"] > consolidation_high) and (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"])) if not pd.isna(consolidation_high) else False

        partial = {
            "SMA20>SMA50": cond_ma,
            "RSI40-60": cond_rsi,
            "Volume>VOL20": cond_vol,
            "Breakout": cond_breakout
        }
        return cond_ma and cond_rsi and cond_vol and cond_breakout, partial

    # Pullback
    elif style == "Pullback":
        cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) else False
        cond_rsi = latest["RSI14"] < 50 if not pd.isna(latest["RSI14"]) else False
        partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
        return cond_pullback and cond_rsi, partial

    # MA Crossover
    elif style == "MA Crossover":
        cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) and not pd.isna(latest["SMA200"]) else False
        partial = {"MA Crossover": cond_ma}
        return cond_ma, partial

    # RSI Range
    elif style == "RSI Range":
        cond_rsi = 40 <= latest["RSI14"] <= 60 if not pd.isna(latest["RSI14"]) else False
        partial = {"RSI40-60": cond_rsi}
        return cond_rsi, partial

    return False, {}

# -----------------------------
# Screener with retry and progress
# -----------------------------
def screen_stocks(tickers: List[str], style: str, retries: int = 3, delay: float = 1.0, **params) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)

    for i, t in enumerate(tickers):
        df = None
        for attempt in range(retries):
            try:
                df = fetch_data(t)
                break
            except Exception as e:
                time.sleep(delay)

        if df is None or df.empty:
            continue

        meets, partial = check_conditions(df, style, params)
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
            "RSI14": round(latest_rsi, 2),
            "SMA20": round(sma20_val, 2),
            "SMA50": round(sma50_val, 2),
            "Volume": int(vol),
            "VOL20": int(vol20),
            "Partial_Match": partial
        })

        progress.progress((i+1)/total)

    return pd.DataFrame(results)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("S&P 500 Multi-Style Screener")

style = st.selectbox("Select style", ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"])
tickers = get_sp500_tickers()  # Scrape Finviz (cached)
st.write(f"Number of tickers fetched: {len(tickers)}")

params = {}
if style == "Momentum + Breakout":
    params["lookback_days"] = st.number_input("Lookback days", value=20, min_value=1)
    params["breakout_buffer"] = st.number_input("Breakout buffer (fraction)", value=0.03, min_value=0.0, max_value=0.1)

if st.button("Run Screener"):
    df_results = screen_stocks(tickers, style, **params)
    if df_results.empty:
        st.error("No stocks met the criteria or data could not be fetched.")
    else:
        st.dataframe(df_results)
