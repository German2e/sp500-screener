import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from typing import List

# -----------------------------
# Static S&P500 Tickers
# -----------------------------
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "XOM", "PFE", "KO", "CVX", "ABBV", "MRK",
    "WMT", "ADBE", "CMCSA", "NFLX", "T", "NKE", "CRM", "PYPL", "INTC", "LLY", "SPGI", "MDT",
    "NEE", "ORCL", "AMGN", "COST", "QCOM", "TXN", "PEP", "BA", "MS", "USB", "GE", "IBM", "RTX",
    "DE", "GS", "UNP", "MMM", "CAT", "LOW", "SYK", "HON", "AXP", "CVS", "CCI", "TMO", "ISRG",
    "ZTS", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY",
    "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO",
    "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA",
    "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT",
    "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE",
    "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO",
    "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK",
    "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO",
    "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT",
    "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD",
    "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP",
    "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL",
    "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT",
    "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK",
    "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY",
    "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO",
    "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA",
    "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT",
    "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE",
    "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO",
    "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK",
    "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO",
    "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT",
    "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD",
    "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP",
    "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL",
    "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT",
    "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK",
    "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY",
    "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO",
    "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA",
    "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT",
    "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE",
    "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK", "CSCO",
    "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO", "DUK",
    "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT", "MO",
    "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD", "LMT",
    "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP", "PLD",
    "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL", "COP",
    "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT", "CL",
    "COP", "PLD", "LMT", "MO", "DUK", "CSCO", "GE", "MDT", "WBA", "SO", "BMY", "DUK", "LMT",
    "CL", "COP", "PLD", "LMT", "MO", "DU"
]

# -----------------------------
# Moving Averages
# -----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

# -----------------------------
# Fetch Data (with fallback)
# -----------------------------
def fetch_data(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty:
            raise ValueError("Empty Yahoo Finance data")
        return df
    except:
        st.warning(f"Yahoo Finance failed for {ticker}. Finviz backup not implemented yet.")
        return pd.DataFrame()  # placeholder for Finviz fallback

# -----------------------------
# Screening Conditions
# -----------------------------
def check_momentum(df: pd.DataFrame, crossover_days: int = 7) -> bool:
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]

    # Trend follower conditions
    cond_price = price > sma20
    cond_ma = sma20 > sma50
    # 20MA crossed 50MA in last 'crossover_days'
    recent_cross = df["SMA20"].iloc[-crossover_days:] > df["SMA50"].iloc[-crossover_days:]
    cond_cross = recent_cross.any()

    # 50MA just crossed 200MA yesterday
    cond_50_200_cross = df["SMA50"].iloc[-2] < df["SMA200"].iloc[-2] and sma50 > sma200

    return cond_price and cond_ma and cond_cross and cond_50_200_cross

def check_breakout(df: pd.DataFrame) -> bool:
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]

    # Price dropped under 20MA but consolidation
    recent_max = df["Close"].iloc[-20:-1].max()
    cond_price = price < sma20 and price > recent_max * 0.95
    cond_ma = sma20 < sma50
    return cond_price and cond_ma

def check_pullback(df: pd.DataFrame) -> bool:
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]

    # Recovery screener
    cond_prev_downtrend = df["Close"].iloc[-10] < df["SMA20"].iloc[-10] and df["SMA20"].iloc[-10] < df["SMA50"].iloc[-10]
    cond_recovery = price > sma20 and sma20 > sma50 and sma50 < sma200
    return cond_prev_downtrend and cond_recovery

# -----------------------------
# Candlestick Plot
# -----------------------------
def plot_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                        open=df['Open'],
                                        high=df['High'],
                                        low=df['Low'],
                                        close=df['Close'],
                                        name=ticker)])
    fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 20), line=dict(color='blue', width=1), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 50), line=dict(color='orange', width=1), name='SMA50'))
    fig.add_trace(go.Scatter(x=df.index, y=sma(df["Close"], 200), line=dict(color='red', width=1), name='SMA200'))
    fig.update_layout(height=400, width=800, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Multi-Style Stock Screener (S&P500)")

strategy = st.sidebar.selectbox("Select Strategy", ["Momentum", "Breakout", "Pullback"])
crossover_days = st.sidebar.slider("Momentum Lookback Days", 1, 14, 7)

if st.sidebar.button("Run Screener"):
    results = []
    progress = st.progress(0)
    total = len(SP500_TICKERS)

    for i, ticker in enumerate(SP500_TICKERS):
        df = fetch_data(ticker)
        if df.empty:
            continue
        try:
            match = False
            if strategy == "Momentum":
                match = check_momentum(df, crossover_days)
            elif strategy == "Breakout":
                match = check_breakout(df)
            elif strategy == "Pullback":
                match = check_pullback(df)

            if match:
                results.append({"Ticker": ticker, "Price": df["Close"].iloc[-1]})
        except Exception as e:
            st.warning(f"Skipping {ticker}: {e}")
        progress.progress((i+1)/total)

    df_results = pd.DataFrame(results)
    st.success(f"Found {len(df_results)} matches!")
    st.dataframe(df_results)

    if not df_results.empty:
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Matches as CSV",
            data=csv,
            file_name=f"{strategy}_matches.csv",
            mime="text/csv"
        )

        # Mini charts in tabs
        tabs = st.tabs(df_results["Ticker"].tolist())
        for idx, ticker in enumerate(df_results["Ticker"]):
            with tabs[idx]:
                df_chart = fetch_data(ticker)
                plot_chart(df_chart, ticker)
