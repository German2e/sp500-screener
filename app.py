import streamlit as st
import yfinance as yf
import pandas as pd
import math

# ----------------------------
# Preloaded S&P 500 tickers
# ----------------------------
SP500_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","BRK.B","JPM","JNJ","V","PG",
    "XOM","UNH","HD","LLY","MRK","AVGO","PEP","KO","ABBV","PFE","MA","CVX","TMO",
    "COST","BAC","DIS","MCD","CSCO","ABT","VZ","ADBE","CRM","NFLX","WMT","INTC",
    "ORCL","ACN","TXN","NKE","HON","LIN","PM","AMGN","UPS","NEE","LOW","MS","GS",
    "BLK","CAT","AMD","RTX","QCOM","IBM","GE","MDT","BA","CVS","DE","NOW","ISRG",
    "AMT","SPGI","LMT","PLD","ADP","MO","BKNG","T","EL","CCI","SYK","INTU","GILD",
    "MDLZ","ZTS","C","USB","CB","SCHW","SO","DUK","PNC","CL","MMM","BDX","CSX",
    "SHW","ETN","CI","WM","NSC","APD","EMR","ICE","EW","MMC","COF","FDX","ITW",
    "HUM","GD","TGT","DHR","REGN","PSA","AON","FIS","MU","CME","MPC","SLB","HCA",
    "PGR","KMB","ECL","AEP","SBUX","MAR","AFL","F","OXY","D","KMI","ALL","MET",
    "PRU","TRV","AIG","A","ROP","MSCI","EXC","TEL","LRCX","KLAC","ORLY","ROST",
    "PAYX","MCK","VLO","STZ","GIS","ADM","CTAS","NOC","DXCM","CMG","ATVI","EA",
    "LHX","IDXX","CDNS","ANET","ODFL","FTNT","CTSH","SNPS","MNST","KDP","HSY",
    "KR","YUM","CMI","PCAR","DLR","SBAC","WELL","SPG","EOG","PSX","HAL","DVN",
    "HES","O","EXR","VTR","WMB","OKE","AEE","PEG","ED","XEL","ES","WEC","AWK",
    "DTE","LUV","DAL","UAL","AAL","NEM","RCL","CCL","NUE","STLD","MOS","CF","ADM",
    # (trimmed for brevity, but we can expand to full 500)
]

# ----------------------------
# Indicators calculator
# ----------------------------
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, progress=False)
        if df.empty:
            return None
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        df["VolumeAvg20"] = df["Volume"].rolling(20).mean()
        return df
    except Exception:
        return None

# ----------------------------
# Strategy functions
# ----------------------------
def check_momentum(df, crossover_days=7):
    if df is None or len(df) < 50:
        return False
    price = float(df["Close"].iloc[-1])
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]

    if math.isnan(sma20) or math.isnan(sma50):
        return False

    cond_price = price > sma20 and sma20 > sma50
    cond_cross = (df["SMA20"].iloc[-crossover_days:] > df["SMA50"].iloc[-crossover_days:]).any()
    return bool(cond_price and cond_cross)

def check_breakout(df):
    if df is None or len(df) < 50:
        return False
    price = float(df["Close"].iloc[-1])
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]

    if math.isnan(sma20) or math.isnan(sma50):
        return False

    cond_down = price < sma20 and sma20 > sma50
    cond_consolidate = (df["Close"].iloc[-5:].max() - df["Close"].iloc[-5:].min()) < 0.05 * price
    return bool(cond_down and cond_consolidate)

def check_pullback(df):
    if df is None or len(df) < 200:
        return False
    price = float(df["Close"].iloc[-1])
    sma20 = df["SMA20"].iloc[-1]
    sma50 = df["SMA50"].iloc[-1]
    sma200 = df["SMA200"].iloc[-1]

    if math.isnan(sma20) or math.isnan(sma50) or math.isnan(sma200):
        return False

    cond_downtrend = price < sma20 and sma20 < sma50 and sma50 < sma200
    cond_recovery = price > sma20 and sma20 > sma50 and price < sma200
    return bool(cond_recovery and not cond_downtrend)

# ----------------------------
# Screener
# ----------------------------
def screen_stocks(tickers, strategy="Momentum", crossover_days=7):
    results = []
    for ticker in tickers:
        df = get_stock_data(ticker)
        if df is None:
            continue
        try:
            if strategy == "Momentum" and check_momentum(df, crossover_days):
                results.append(ticker)
            elif strategy == "Breakout" and check_breakout(df):
                results.append(ticker)
            elif strategy == "Pullback" and check_pullback(df):
                results.append(ticker)
        except Exception:
            continue
    return results

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📈 S&P 500 Stock Screener")
st.sidebar.header("Screener Options")

strategy = st.sidebar.selectbox("Select Strategy", ["Momentum", "Breakout", "Pullback"])
crossover_days = st.sidebar.slider("Momentum crossover days", 3, 14, 7)

st.write(f"### Running {strategy} screener on S&P 500...")

matches = screen_stocks(SP500_TICKERS, strategy, crossover_days)

if matches:
    st.success(f"Found {len(matches)} matching stocks!")
    st.write(pd.DataFrame(matches, columns=["Ticker"]))
else:
    st.warning("⚠️ No matching stocks found. Try adjusting filters or timeframe.")




