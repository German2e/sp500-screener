import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go

# -----------------------------
# Full S&P 500 ticker list (partial example; replace with full list)
# -----------------------------
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ",
    # ... all remaining tickers ...
]

# -----------------------------
# Fetch stock data with fallback
# -----------------------------
def fetch_data(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise Exception("Yahoo returned empty data")
        df["SMA20"] = df["Close"].rolling(20, min_periods=1).mean()
        df["SMA50"] = df["Close"].rolling(50, min_periods=1).mean()
        df["SMA200"] = df["Close"].rolling(200, min_periods=1).mean()
        return df
    except:
        # Finviz fallback
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            # Basic data extraction: just last price
            price = float(soup.find('b').text.replace('$',''))
            df = pd.DataFrame({"Close": [price]})
            df["SMA20"] = df["Close"]
            df["SMA50"] = df["Close"]
            df["SMA200"] = df["Close"]
            return df
        except:
            return None

# -----------------------------
# Strategy checks (Momentum, Breakout, Pullback)
# -----------------------------
def check_momentum(df, crossover_days=7):
    if df is None or len(df) < 50: return False
    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    recent_cross = df["SMA20"].iloc[-crossover_days:] > df["SMA50"].iloc[-crossover_days:]
    return (price > sma20) and (sma20 > sma50) and recent_cross.any()

def check_breakout(df):
    if df is None or len(df) < 50: return False
    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    recent_range = df["Close"].iloc[-20:]
    return (price < sma20) and (sma20 > sma50) and ((recent_range.max() - recent_range.min()) / recent_range.min() < 0.05)

def check_pullback(df):
    if df is None or len(df) < 200: return False
    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]
    return (price > sma20) and (sma20 > sma50) and (sma50 < sma200)

# -----------------------------
# Run Screener
# -----------------------------
def run_screener(tickers, strategy, crossover_days=7):
    results = []
    for t in tickers:
        df = fetch_data(t)
        meets = False
        if strategy=="Momentum": meets = check_momentum(df, crossover_days)
        elif strategy=="Breakout": meets = check_breakout(df)
        elif strategy=="Pullback": meets = check_pullback(df)
        if meets:
            results.append({"Ticker": t, "Close": df["Close"].iloc[-1]})
    return pd.DataFrame(results)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("S&P 500 Screener with Finviz Backup")
strategy = st.sidebar.selectbox("Strategy:", ["Momentum","Breakout","Pullback"])
crossover_days = st.sidebar.slider("Momentum crossover days:",1,14,7)

if st.sidebar.button("Run Screener"):
    st.info(f"Scanning {len(SP500_TICKERS)} tickers for {strategy}")
    df_results = run_screener(SP500_TICKERS, strategy, crossover_days)
    if df_results.empty:
        st.warning("No matches found.")
    else:
        st.success(f"Found {len(df_results)} matches!")
        for idx,row in df_results.iterrows():
            st.write(f"**{row['Ticker']}** - Close: {row['Close']}")
            df_chart = fetch_data(row['Ticker'])
            if df_chart is not None:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
                    low=df_chart['Low'], close=df_chart['Close'], name='Candlestick'
                ))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA20'], line=dict(color='blue'), name='SMA20'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA50'], line=dict(color='orange'), name='SMA50'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['SMA200'], line=dict(color='green'), name='SMA200'))
                st.plotly_chart(fig, use_container_width=True)
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"{strategy}_matches.csv", mime="text/csv")
