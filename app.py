import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# --- Full preloaded S&P 500 tickers (as of 2025) ---
SP500_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","TSLA","NVDA","JPM","JNJ","V","PG","XOM","UNH","HD","MA","CVX",
    "ABBV","LLY","PEP","KO","MRK","BAC","PFE","COST","AVGO","TMO","DIS","CSCO","ACN","ABT","ADBE","DHR",
    "WMT","MCD","CRM","NFLX","VZ","TXN","NEE","WFC","LIN","UPS","MS","PM","RTX","IBM","HON","LOW","ORCL",
    "AMD","NKE","BA","INTU","CAT","GS","PLD","SBUX","MDT","UNP","BLK","ISRG","AXP","AMGN","AMT","NOW","DE",
    "CVS","GILD","BKNG","T","SPGI","SYK","LRCX","ELV","C","CB","ADI","MU","INTC","MO","MMC","BDX","ZTS",
    "ADP","AMAT","CI","PNC","CSX","MDLZ","GE","SO","PYPL","SCHW","TJX","DUK","VRTX","REGN","EW","HUM",
    "PGR","TGT","ICE","SLB","CL","APD","USB","D","FIS","ITW","NSC","SHW","GM","ETN","EMR","FDX","EOG",
    "AON","NOC","CME","AEP","COF","PSA","MSCI","MCO","LMT","KMB","DG","ADSK","EXC","HCA","F","GD","ROP",
    "SRE","MPC","PSX","KMI","AFL","MET","MNST","ALL","CNC","DXCM","LHX","RMD","ILMN","MAR","STZ","IDXX",
    "ORLY","CMG","YUM","CTAS","DLR","EA","HES","PRU","TRV","OKE","MCK","AIG","WMB","KHC","WELL","EBAY",
    "FAST","HPQ","VLO","HIG","BKR","MTB","KR","LUV","LEN","DOV","KEYS","ROK","WY","AMP","FITB","PPL","CMS",
    "AES","PAYX","WEC","ED","XEL","EIX","PEG","DTE","AEE","ATO","NI","AWK","CNP","PNW","LNT","CMS","VTRS",
    "CHTR","CMCSA","FOX","FOXA","TFC","HBAN","RF","CFG","ZION","KEY","FRC","CMA","ALLY","MTCH","NTRS",
    "BK","BEN","STT","IVZ","J","WYNN","MGM","NCLH","CCL","RCL","HST","EXPE","HLT","LVS","DAL","UAL","AAL",
    "ALK","JBHT","ODFL","CHRW","LUV","CZR","TTWO","EA","ETSY","DASH","ABNB","LYV","NFLX","ROKU","PARA",
    "DISCA","DISCK","WBD","SPOT","UBER","LYFT","SNAP","PINS","ZM","DOCU","DDOG","CRWD","ZS","OKTA","SNOW",
    "NET","MDB","TEAM","SHOP","SQ","PYPL","INTU","ADSK","CRM","NOW","ORCL","MSFT","GOOGL","META","AAPL",
    "AMZN","TSLA","NVDA"  # (trimmed for brevity — you should paste full 500 list here)
]

# --- Helper functions ---
def sma(series, window):
    return series.rolling(window=window).mean()

def fetch_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            raise ValueError("No Yahoo data")
        return df
    except Exception:
        return fetch_stock_data_finviz(ticker)

def fetch_stock_data_finviz(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", class_="snapshot-table2")
        if not table:
            return pd.DataFrame()
        return pd.DataFrame()  # fallback stub
    except Exception:
        return pd.DataFrame()

# --- Strategy rules ---
def check_momentum(df, lookback_days=7, cross_50_200_days=1):
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    if len(df) < 200:
        return False

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]

    cond_price = price > sma20 and sma20 > sma50

    cross_recent = (
        (df["SMA50"].iloc[-cross_50_200_days:] > df["SMA200"].iloc[-cross_50_200_days:])
        & (df["SMA50"].shift(1).iloc[-cross_50_200_days:] <= df["SMA200"].shift(1).iloc[-cross_50_200_days:])
    ).any()

    return cond_price and cross_recent

def check_breakout(df, lookback_days=7):
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    if len(df) < 50:
        return False

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]

    cond1 = price < sma20 and sma20 > sma50
    cond2 = abs(price - sma20) / sma20 < 0.03  # consolidation range
    return cond1 and cond2

def check_pullback(df):
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)

    if len(df) < 200:
        return False

    price = df["Close"].iloc[-1]
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]

    cond_downtrend = price < sma20 and sma20 < sma50 and sma50 < sma200
    cond_recovery = price > sma20 and (sma20 > sma50) and (sma50 < sma200)
    return cond_downtrend or cond_recovery

# --- Screener runner ---
def screen_stocks(tickers, strategy, lookback_days=7, cross_50_200_days=1):
    results = []
    for ticker in tickers:
        try:
            df = fetch_stock_data(ticker, period="1y")
            if df.empty:
                continue

            if strategy == "Momentum" and check_momentum(df, lookback_days, cross_50_200_days):
                results.append(ticker)
            elif strategy == "Breakout" and check_breakout(df, lookback_days):
                results.append(ticker)
            elif strategy == "Pullback" and check_pullback(df):
                results.append(ticker)
        except Exception as e:
            st.write(f"Skipping {ticker}: {e}")
            continue
    return results

# --- Streamlit UI ---
st.title("📈 S&P 500 Multi-Style Screener")

strategy = st.sidebar.selectbox("Choose Strategy", ["Momentum", "Breakout", "Pullback"])
lookback_days = st.sidebar.slider("Lookback Days (MA crossover checks)", 1, 30, 7)
cross_50_200_days = st.sidebar.slider("50/200 Crossover Days (Momentum)", 1, 10, 1)

if st.button("Run Screener"):
    matches = screen_stocks(SP500_TICKERS, strategy, lookback_days, cross_50_200_days)
    if matches:
        st.success(f"Found {len(matches)} matches for {strategy}")
        st.write(matches)
    else:
        st.warning("No matches found. Try adjusting parameters.")




