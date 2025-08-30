import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers (Wiki + Finviz fallback)
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> List[str]:
    tickers = []

    # --- Try Wikipedia first ---
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table:
            for row in table.findAll("tr")[1:]:
                ticker = row.findAll("td")[0].text.strip().replace(".", "-")
                tickers.append(ticker)
    except Exception as e:
        st.warning(f"Wikipedia fetch failed: {e}")

    # --- Fallback to Finviz if Wiki fails ---
    if not tickers:
        try:
            url = "https://finviz.com/screener.ashx?v=111&f=idx_sp500"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            rows = soup.find_all("a", class_="screener-link-primary")
            tickers = [r.text.strip() for r in rows if r.text.isupper()]
        except Exception as e:
            st.error(f"Finviz fetch failed: {e}")

    return sorted(list(set(tickers)))


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
# Condition checkers
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> bool:
    df = df.copy()
    df["SMA20"] = sma(df["Close"], params.get("sma20", 20))
    df["SMA50"] = sma(df["Close"], params.get("sma50", 50))
    df["SMA200"] = sma(df["Close"], params.get("sma200", 200))
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL20"] = sma(df["Volume"], 20)

    if df.shape[0] < max(50, params.get("lookback_days", 30) + 1):
        return False

    latest = df.iloc[-1]

    if style == "Momentum + Breakout":
        cond_ma = bool(latest["SMA20"] > latest["SMA50"])
        cond_rsi = bool(40 <= latest["RSI14"] <= 60)
        cond_vol = bool(latest["Volume"] > latest["VOL20"])

        consolidation_window = df["Close"].iloc[-(params["lookback_days"]+1):-1].dropna()
        if consolidation_window.empty:
            return False
        consolidation_high = consolidation_window.max()

        cond_breakout = bool(
            (latest["Close"] > consolidation_high) and 
            (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"]))
        )
        return cond_ma and cond_rsi and cond_vol and cond_breakout

    elif style == "Pullback":
        cond_pullback = bool(latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"])
        cond_rsi = bool(latest["RSI14"] < 50)
        return cond_pullback and cond_rsi

    elif style == "MA Crossover":
        cond_ma = bool(latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"])
        return cond_ma

    elif style == "RSI Range":
        return bool(40 <= latest["RSI14"] <= 60)

    return False


# -----------------------------
# Main screener
# -----------------------------
def screen_stocks(tickers: List[str], style: str, **params) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers):
        try:
            df = fetch_data(t)
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
        except Exception as e:
            print(f"Skipping {t}: {e}")
        progress.progress((i+1)/total)
    return pd.DataFrame(results)


# -----------------------------
# Streamlit interface
# -----------------------------
st.title("Multi-Style Stock Screener (S&P500)")

style = st.sidebar.selectbox("Select Trading Style:", ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"])

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

        st.error("No tickers available. Check Wikipedia/Finviz source or network.")

    else:

        st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")

        df_results = screen_stocks(tickers, style, **params)



        if df_results.empty or "Meets_Entry" not in df_results.columns:

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

                    file_name=f"{style.replace(' ', '_')}_matches.csv",

                    mime="text/csv"

                )


