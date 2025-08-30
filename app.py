import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers from Wikipedia
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            table = soup.find("table", {"class": "wikitable"})
            if table is None:
                st.error("Could not find S&P500 table on Wikipedia.")
                return []
        tickers = []
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if cols:
                ticker = cols[0].text.strip().replace(".", "-")
                tickers.append(ticker)
        return tickers
    except Exception as e:
        st.error(f"Failed to fetch S&P500 tickers: {e}")
        return []

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
# Fetch data
# -----------------------------
def fetch_data(ticker: str, period: str = "600d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

# -----------------------------
# Condition checkers (returns partial info for debug)
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> (bool, dict):
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
        cond_ma = not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) and latest["SMA20"] > latest["SMA50"]
        cond_rsi = not pd.isna(latest["RSI14"]) and 40 <= latest["RSI14"] <= 60
        cond_vol = not pd.isna(latest["VOL20"]) and latest["Volume"] > latest["VOL20"]

        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        if pd.isna(consolidation_high):
            cond_breakout = False
        else:
            cond_breakout = (latest["Close"] > consolidation_high) and \
                             (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"]))

        partial = {
            "SMA20>SMA50": cond_ma,
            "RSI40-60": cond_rsi,
            "Volume>VOL20": cond_vol,
            "Breakout": cond_breakout
        }

        return bool(cond_ma and cond_rsi and cond_vol and cond_breakout), partial

    elif style == "Pullback":
        cond_pullback = not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) and latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"]
        cond_rsi = not pd.isna(latest["RSI14"]) and latest["RSI14"] < 50
        partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
        return bool(cond_pullback and cond_rsi), partial

    elif style == "MA Crossover":
        cond_ma = not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) and not pd.isna(latest["SMA200"]) and latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"]
        partial = {"MA Crossover": cond_ma}
        return cond_ma, partial

    elif style == "RSI Range":
        cond_rsi = not pd.isna(latest["RSI14"]) and 40 <= latest["RSI14"] <= 60
        partial = {"RSI40-60": cond_rsi}
        return cond_rsi, partial

    return False, partial

# -----------------------------
# Main screener
# -----------------------------
def screen_stocks(tickers: List[str], style: str, debug: bool = False, **params) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers):
        try:
            df = fetch_data(t)
            meets, partial = check_conditions(df, style, params)
            latest_close = df["Close"].iloc[-1]
            latest_rsi = rsi(df["Close"]).iloc[-1]
            sma20_val = sma(df["Close"], 20).iloc[-1]
            sma50_val = sma(df["Close"], 50).iloc[-1]
            vol = df["Volume"].iloc[-1]
            vol20 = sma(df["Volume"], 20).iloc[-1]

            result = {
                "Ticker": t,
                "Meets_Entry": meets,
                "Close": round(latest_close, 2),
                "RSI14": round(latest_rsi, 2) if not np.isnan(latest_rsi) else None,
                "SMA20": round(sma20_val, 2) if not np.isnan(sma20_val) else None,
                "SMA50": round(sma50_val, 2) if not np.isnan(sma50_val) else None,
                "Volume": int(vol),
                "VOL20": int(vol20) if not np.isnan(vol20) else None
            }

            if debug:
                result["Debug"] = partial

            results.append(result)

        except Exception as e:
            st.warning(f"Skipping {t}: {e}")

        if total > 0:
            progress.progress((i+1)/total)

    return pd.DataFrame(results)

# -----------------------------
# Streamlit interface
# -----------------------------
st.title("Multi-Style Stock Screener (S&P500)")

style = st.sidebar.selectbox("Select Trading Style:", ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"])
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)

# Style parameters
params = {}
params["lookback_days"] = st.sidebar.slider("Lookback Days (Momentum/Breakout)", 10, 60, 30)
params["breakout_buffer"] = st.sidebar.slider("Breakout Buffer (%)", 0.01, 0.10, 0.05)
params["sma20"] = st.sidebar.slider("SMA20", 10, 50, 20)
params["sma50"] = st.sidebar.slider("SMA50", 20, 100, 50)
params["sma200"] = st.sidebar.slider("SMA200", 50, 250, 200)

# Run screener
if st.sidebar.button("Run Screener"):
    tickers = get_sp500_tickers()
    if not tickers:
        st.error("No tickers to scan. Check Wikipedia source or network connection.")
    else:
        st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")
        df_results = screen_stocks(tickers, style, debug=debug_mode, **params)
        matches = df_results[df_results["Meets_Entry"]]

        st.success(f"Found {len(matches)} matches!")
        st.dataframe(df_results if debug_mode else matches)

        if not matches.empty:
            csv = matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matches as CSV",
                data=csv,
                file_name=f"{style.replace(' ', '_')}_matches.csv",
                mime="text/csv"
            )
