import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers from Wikipedia (with User-Agent to avoid 403)
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
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
# Condition checkers (safe boolean checks)
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

    # Momentum + Breakout
    if style == "Momentum + Breakout":
        sma20_val = latest["SMA20"]
        sma50_val = latest["SMA50"]
        rsi_val = latest["RSI14"]
        vol_val = latest["VOL20"]

        if pd.isna(sma20_val) or pd.isna(sma50_val) or pd.isna(rsi_val) or pd.isna(vol_val):
            return False, {}

        cond_ma = float(sma20_val) > float(sma50_val)
        cond_rsi = 40 <= float(rsi_val) <= 60
        cond_vol = latest["Volume"] > float(vol_val)

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

    # Pullback
    elif style == "Pullback":
        sma20_val = latest["SMA20"]
        sma50_val = latest["SMA50"]
        rsi_val = latest["RSI14"]

        if pd.isna(sma20_val) or pd.isna(sma50_val) or pd.isna(rsi_val):
            return False, {}

        cond_pullback = latest["Close"] < float(sma20_val) and latest["Close"] > float(sma50_val)
        cond_rsi = rsi_val < 50

        partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
        return bool(cond_pullback and cond_rsi), partial

    # MA Crossover
    elif style == "MA Crossover":
        sma20_val = latest["SMA20"]
        sma50_val = latest["SMA50"]
        sma200_val = latest["SMA200"]

        if pd.isna(sma20_val) or pd.isna(sma50_val) or pd.isna(sma200_val):
            return False, {}

        cond_ma = float(sma20_val) > float(sma50_val) and float(sma50_val) > float(sma200_val)
        partial = {"MA Crossover": cond_ma}
        return cond_ma, partial

    # RSI Range
    elif style == "RSI Range":
        rsi_val = latest["RSI14"]
        if pd.isna(rsi_val):
            return False, {}
        cond_rsi = 40 <= float(rsi_val) <= 60
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
