import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers (fixed list)
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> list[str]:
    tickers = [
        "MMM", "AOS", "ABT", "ABBV", "ACN", "ATVI", "AYI", "ADBE", "AAP", "AMD",
        "AES", "AFL", "A", "APD", "AKAM", "ALK", "ALL", "GOOGL", "GOOG", "MO",
        "AMZN", "AMCR", "AEE", "AAL", "APA", "AAPL", "AMAT", "APTV", "ADM", "ARNC",
        "AJG", "AIZ", "T", "ADSK", "ADP", "AZO", "AVB", "AVGO", "BA", "BKR", "BLL",
        "BAC", "BBY", "BDX", "BRK-B", "BBWI", "BIO", "BIIB", "BLK", "BAH", "BWA",
        "BSX", "BMY", "BR", "CHRW", "COG", "CDW", "CPB", "COF", "CAH", "KMX", "CCL",
        "CAT", "CBOE", "CBRE", "CDNS", "CE", "CNC", "CNP", "CTAS", "CSCO", "C", "CFG",
        "CTVA", "CLX", "CME", "CMS", "KO", "CTSH", "CL", "CMG", "CB", "CHD", "CHTR",
        "CVX", "CMI", "COST", "CCI", "CSX", "CARR", "CTVA", "CME", "CVS", "DHI", "DHR",
        "DRI", "DOV", "DOW", "DTE", "DUK", "DXCM", "EMR", "ETN", "EBAY", "ECL", "EIX",
        "EW", "EA", "LLY", "EMN", "ETR", "EVRG", "ES", "EXC", "EXPD", "EXR", "FFIV",
        "FIS", "FISV", "FLT", "FMC", "F", "FTNT", "FTV", "GD", "GE", "GIS", "GM", "GOOG",
        "GPC", "GILD", "GLW", "GS", "HAL", "HBI", "HOG", "HCA", "PEAK", "HSY", "HES",
        "HPE", "HUM", "HON", "HRL", "HST", "HII", "IBM", "IDXX", "ILMN", "INCY", "INTC",
        "ICE", "IP", "IPG", "IQV", "IRM", "J", "JBHT", "JNJ", "JPM", "K", "KEYS", "KMB",
        "KIM", "KMI", "KLAC", "KSS", "KHC", "LHX", "LH", "LMT", "L", "LOW", "LIN", "LYB",
        "LNC", "LVS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MPWR",
        "MS", "MSFT", "MAA", "NDAQ", "NEM", "NFLX", "NOC", "NUE", "NVDA", "NVR", "ORLY",
        "OXY", "ODFL", "OMC", "OKE", "ORCL", "PCAR", "PEP", "PKI", "PM", "PNC", "POOL",
        "PPG", "PPL", "PFE", "PSA", "PH", "PYPL", "PNR", "PXD", "QRVO", "QCOM", "DGX",
        "RCL", "RTX", "REG", "REGN", "RF", "RSG", "RMD", "ROP", "ROST", "RHI", "SPGI",
        "CRM", "SLB", "SNA", "SO", "LUV", "SWK", "SYY", "SPG", "SYF", "SNPS", "SIVB",
        "SWKS", "SBUX", "STT", "STE", "SYK", "SIVB", "SYY", "TFC", "TMO", "TJX", "TSLA",
        "TXN", "TDG", "TRV", "TFC", "TGT", "TROW", "TT", "TDY", "UNH", "UPS", "URI", "UTX",
        "UHS", "VFC", "VLO", "VMC", "VZ", "VRTX", "V", "WMT", "WBA", "WM", "WBD", "WDC",
        "WEC", "WFC", "WELL", "WMB", "WRB", "WU", "WY", "XEL", "XYL", "YUM", "ZBRA", "ZTS"
    ]
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
# Fetch data
# -----------------------------
def fetch_data(ticker: str, period: str = "240d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

# -----------------------------
# Condition checkers for each style
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> bool:
    df = df.copy()
    df["SMA20"] = sma(df["Close"], params.get("sma20", 20))
    df["SMA50"] = sma(df["Close"], params.get("sma50", 50))
    df["SMA200"] = sma(df["Close"], params.get("sma200", 200))
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL20"] = sma(df["Volume"], 20)

    if df.shape[0] < max(50, params.get("lookback_days", 30)+1):
        return False

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
    for t in tickers:
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
    st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")

    # Run the screener
    df_results = screen_stocks(tickers, style, **params)

    # ---- DEBUG LOGGING ----
    if not df_results.empty:
        st.subheader("Sample of fetched data (first 5 tickers):")
        st.dataframe(df_results.head(5))

        st.write(f"Columns fetched: {df_results.columns.tolist()}")
        st.write(f"Total tickers scanned: {len(df_results)}")

        # Show statistics on Meets_Entry column if exists
        if "Meets_Entry" in df_results.columns:
            st.write(df_results["Meets_Entry"].value_counts())
    else:
        st.warning("No data returned for tickers — check your fetch_data function or parameters.")

    # ---- FILTER MATCHES ----
    if not df_results.empty and "Meets_Entry" in df_results.columns:
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
    else:
        st.warning("No results found — try adjusting your parameters.")

