import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List

# -----------------------------
# Preloaded S&P 500 tickers
# -----------------------------
SP500_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","GOOG","TSLA","BRK.B","NVDA","META","UNH",
    "JNJ","V","JPM","PG","HD","MA","DIS","PYPL","BAC","ADBE","CMCSA","NFLX",
    # … add all 500 tickers here …
]

# -----------------------------
# Indicators
# -----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    if len(series) < window:
        return pd.Series([np.nan]*len(series))
    return series.rolling(window=window, min_periods=window).mean()

# -----------------------------
# Fetch data safely
# -----------------------------
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        df["SMA20"] = sma(df["Close"], 20)
        df["SMA50"] = sma(df["Close"], 50)
        df["SMA200"] = sma(df["Close"], 200)
        return df
    except Exception as e:
        return None

# -----------------------------
# Strategy checks
# -----------------------------
def check_momentum(df: pd.DataFrame, crossover_days: int = 7) -> bool:
    if df is None or len(df) < 50:
        return False
    recent = df.tail(crossover_days)
    # Price above 20MA
    cond_price = df["Close"].iloc[-1] > df["SMA20"].iloc[-1]
    # 20MA above 50MA
    cond_ma = df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]
    # 20MA cut above 50MA in last X days
    cross = (recent["SMA20"] > recent["SMA50"]) & (recent["SMA20"].shift(1) <= recent["SMA50"].shift(1))
    cond_cross = cross.any()
    return cond_price and cond_ma and cond_cross

def check_breakout(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 50:
        return False
    recent = df.tail(20)
    # Price under 20MA
    cond_price = df["Close"].iloc[-1] < df["SMA20"].iloc[-1]
    # 20MA moving back to 50MA (approaching from below)
    cond_ma = df["SMA20"].iloc[-1] < df["SMA50"].iloc[-1] and df["SMA20"].iloc[-2] < df["SMA50"].iloc[-2]
    # Price consolidating (within 5% range)
    range_pct = (recent["Close"].max() - recent["Close"].min()) / recent["Close"].min()
    cond_range = range_pct <= 0.05
    return cond_price and cond_ma and cond_range

def check_pullback(df: pd.DataFrame) -> bool:
    if df is None or len(df) < 50:
        return False
    # Downtrend: 20<50<200
    downtrend = df["SMA20"].iloc[-1] < df["SMA50"].iloc[-1] < df["SMA200"].iloc[-1]
    # Price bottomed and now above 20MA
    recovery = df["Close"].iloc[-1] > df["SMA20"].iloc[-1] and df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]
    return downtrend and recovery

# -----------------------------
# Screen stocks
# -----------------------------
def screen_stocks(tickers: List[str], strategy: str, crossover_days: int = 7) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers):
        df = fetch_data(t)
        meets = False
        if strategy == "Momentum":
            meets = check_momentum(df, crossover_days)
        elif strategy == "Breakout":
            meets = check_breakout(df)
        elif strategy == "Pullback":
            meets = check_pullback(df)
        if df is not None:
            results.append({
                "Ticker": t,
                "Meets_Entry": meets,
                "Close": round(df["Close"].iloc[-1], 2),
                "SMA20": round(df["SMA20"].iloc[-1], 2) if not np.isnan(df["SMA20"].iloc[-1]) else None,
                "SMA50": round(df["SMA50"].iloc[-1], 2) if not np.isnan(df["SMA50"].iloc[-1]) else None,
                "SMA200": round(df["SMA200"].iloc[-1], 2) if not np.isnan(df["SMA200"].iloc[-1]) else None,
            })
        progress.progress((i+1)/total)
    return pd.DataFrame(results)

# -----------------------------
# Streamlit interface
# -----------------------------
st.title("S&P 500 Multi-Style Screener")

strategy = st.sidebar.selectbox("Select Strategy:", ["Momentum", "Breakout", "Pullback"])
crossover_days = st.sidebar.slider("Momentum Crossover Window (days)", 3, 14, 7)

if st.sidebar.button("Run Screener"):
    st.info(f"Scanning {len(SP500_TICKERS)} tickers for {strategy} strategy...")
    df_results = screen_stocks(SP500_TICKERS, strategy, crossover_days)
    matches = df_results[df_results["Meets_Entry"]]
    st.success(f"Found {len(matches)} matches!")
    st.dataframe(matches)

    if not matches.empty:
        csv = matches.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Matches as CSV",
            data=csv,
            file_name=f"{strategy.replace(' ', '_')}_matches.csv",
            mime="text/csv"
        )

