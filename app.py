import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- Preloaded S&P 500 tickers (shortened here for space, load full list in your code)
SP500_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "PG",
    "UNH", "HD", "MA", "CVX", "XOM", "LLY", "PFE", "ABBV", "MRK", "KO",
    # ... include all 500 tickers
]

# --- Helpers
def safe_sma(df, col, window):
    if col not in df:
        return pd.Series([np.nan] * len(df), index=df.index)
    return df[col].rolling(window=window, min_periods=1).mean()

def fetch_data(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, progress=False, threads=False)
        if df.empty:
            return None
        return df
    except Exception:
        return None

# --- Strategies
def check_momentum(df, lookback_days=7) -> bool:
    if df is None or df.shape[0] < 60:
        return False
    df["SMA20"] = safe_sma(df, "Close", 20)
    df["SMA50"] = safe_sma(df, "Close", 50)

    if pd.isna(df["SMA20"].iloc[-1]) or pd.isna(df["SMA50"].iloc[-1]):
        return False

    cond_price = df["Close"].iloc[-1] > df["SMA20"].iloc[-1]
    cond_ma = df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]

    recent = df.iloc[-lookback_days:]
    cross = ((recent["SMA20"] > recent["SMA50"]) &
             (recent["SMA20"].shift(1) <= recent["SMA50"].shift(1))).any()

    return bool(cond_price and cond_ma and cross)

def check_breakout(df, lookback_days=7) -> bool:
    if df is None or df.shape[0] < 60:
        return False
    df["SMA20"] = safe_sma(df, "Close", 20)
    df["SMA50"] = safe_sma(df, "Close", 50)

    if pd.isna(df["SMA20"].iloc[-1]) or pd.isna(df["SMA50"].iloc[-1]):
        return False

    cond1 = df["Close"].iloc[-1] < df["SMA20"].iloc[-1]  # price dipped below 20MA
    cond2 = df["SMA20"].iloc[-1] > df["SMA50"].iloc[-1]  # 20MA still > 50MA

    recent = df.iloc[-lookback_days:]
    consolidating = (recent["Close"].max() - recent["Close"].min()) / recent["Close"].mean() < 0.05

    return bool(cond1 and cond2 and consolidating)

def check_pullback(df, lookback_days=7) -> bool:
    if df is None or df.shape[0] < 200:
        return False
    df["SMA20"] = safe_sma(df, "Close", 20)
    df["SMA50"] = safe_sma(df, "Close", 50)
    df["SMA200"] = safe_sma(df, "Close", 200)

    if pd.isna(df["SMA20"].iloc[-1]) or pd.isna(df["SMA50"].iloc[-1]) or pd.isna(df["SMA200"].iloc[-1]):
        return False

    cond_downtrend = (
        df["Close"].iloc[-1] < df["SMA20"].iloc[-1] and
        df["SMA20"].iloc[-1] < df["SMA50"].iloc[-1] and
        df["SMA50"].iloc[-1] < df["SMA200"].iloc[-1]
    )

    cond_recovery = (
        df["Close"].iloc[-1] > df["SMA20"].iloc[-1] and
        ((df["SMA20"] > df["SMA50"]) &
         (df["SMA20"].shift(1) <= df["SMA50"].shift(1))).any()
    )

    return bool(cond_recovery and not cond_downtrend)

# --- Screening logic
def screen_stocks(tickers, strategy, lookback_days=7):
    results = []
    for ticker in tickers:
        df = fetch_data(ticker)
        if df is None:
            results.append({"Ticker": ticker, "Meets_Entry": False, "Notes": "Data unavailable"})
            continue
        try:
            if strategy == "Momentum":
                meets = check_momentum(df, lookback_days)
            elif strategy == "Breakout":
                meets = check_breakout(df, lookback_days)
            elif strategy == "Pullback":
                meets = check_pullback(df, lookback_days)
            else:
                meets = False
            results.append({
                "Ticker": ticker,
                "Price": round(df["Close"].iloc[-1], 2),
                "Meets_Entry": meets,
                "Notes": ""
            })
        except Exception as e:
            results.append({"Ticker": ticker, "Meets_Entry": False, "Notes": str(e)})
    return pd.DataFrame(results)

# --- Streamlit UI
st.title("📈 S&P 500 Screener (Momentum, Breakout, Pullback)")

strategy = st.selectbox("Choose strategy:", ["Momentum", "Breakout", "Pullback"])
lookback_days = st.slider("Lookback days for signals", 3, 30, 7)

if st.button("Run Screener"):
    with st.spinner("Screening stocks..."):
        df_results = screen_stocks(SP500_TICKERS, strategy, lookback_days)

    if not df_results.empty:
        matches = df_results[df_results["Meets_Entry"]]
        st.subheader(f"✅ Matches found: {len(matches)}")
        st.dataframe(matches)
    else:
        st.warning("No valid stock data returned.")


