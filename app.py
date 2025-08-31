import streamlit as st
import yfinance as yf
import pandas as pd
import datetime

# -----------------------
# Preloaded S&P 500 tickers (2025 list)
# -----------------------
SP500_TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK.B","LLY","AVGO","JPM",
    "TSLA","V","UNH","PG","MA","XOM","HD","JNJ","MRK","COST",
    "ABBV","NFLX","CRM","WMT","CVX","ADBE","BAC","KO","PEP","ORCL",
    "PFE","TMO","CSCO","ACN","ABT","DHR","LIN","MCD","AMD","DIS",
    "VZ","WFC","INTU","PM","CAT","BMY","TXN","NEE","UNP","NKE",
    "HON","IBM","GS","COP","AMGN","LOW","RTX","MS","GE","UPS",
    "INTC","LMT","SPGI","QCOM","DE","AXP","PLD","BLK","SYK","SCHW",
    "CVS","ELV","AMAT","TJX","MDT","T","BKNG","BA","NOW","MDLZ",
    "ISRG","ADI","C","ZTS","GILD","REGN","PGR","PYPL","MMC","MO",
    "ADP","CI","CB","SO","DUK","PNC","SLB","BDX","AMT","CL",
    "ICE","APD","ETN","CSX","VRTX","SHW","MU","CMCSA","EOG","ITW",
    "HUM","NSC","TGT","EQIX","MMM","MPC","FCX","FI","CME","D",
    "FIS","WM","ORLY","AON","PXD","USB","PH","NOC","MAR","EW",
    "PSX","EMR","TRV","HCA","MCK","OXY","MRNA","HES","ROP","LRCX",
    "KMI","AFL","ADSK","GIS","NEM","DVN","NXPI","KMB","MSI","CTAS",
    "AEP","GD","COF","KLAC","MCO","STZ","WELL","PCAR","SRE","AZO",
    "MCHP","IDXX","HLT","JCI","ALL","PAYX","WMB","FICO","PRU","AIG",
    "AMP","FTNT","KHC","CDNS","DXCM","BK","WEC","MET","DOW","MNST",
    "TEL","CNC","O","BIIB","CTVA","CSGP","PSA","CMG","A","RSG",
    "AJG","DD","VLO","MTD","OKE","AME","XEL","IQV","KR","TT",
    "ODFL","WBD","HPQ","PEG","VRSK","MSCI","HIG","EXC","PCG","SYY",
    "SBAC","EA","PPG","SNPS","SPG","DLR","ROST","WAT","LHX","ED",
    "ABC","TSCO","FAST","STT","ANET","F","ECL","GLW","CCI","DFS",
    "BKR","HSY","YUM","CTSH","VICI","CAH","CEG","ON","OTIS","CBRE",
    "CHTR","INVH","NDAQ","BR","MTB","LEN","TRGP","KEYS","ZBH",
    "HPE","EXPE","CINF","EFX","VTRS","FTV","AWK","MLM","CPRT","SWKS",
    "UAL","ILMN","CARR","ETR","WAB","EIX","DG","DDOG","ROK","CMS",
    "GPN","FLT","STE","HWM","NUE","NTRS","GWW","BAX","APTV","IR",
    "BXP","RCL","VMC","LYB","ALB","HOLX","FANG","DAL","ES","DHI",
    "IFF","CFG","KEY","TSN","GL","BALL","FITB","MKC","RF","BBY",
    "EBAY","HBAN","TDG","PWR","DTE","TER","AVB","MPWR","STX","BRO",
    "WTW","PPL","VRSN","K","NTAP","CLX","AKAM","AES","EXPD","ULTA",
    "CAG","GRMN","LH","TYL","PKI","ANSS","CHD","DOV","MKTX","CBOE",
    "CNP","LUV","ZBRA","ETSY","LYV","DPZ","NVR","WRB","IP","SJM",
    "AEE","J","BBWI","RMD","ALGN","GNRC","CF","REG",
    "PTC","DGX","JBHT","MAS","WDC","MTCH","FOX","FOXA","NRG","TRMB",
    "VTR","POOL","VFC","APA","AAL","WY","KIM","LVS","HST","MGM",
    "OZK","PNW","HAS","UHS","NCLH","TAP","FRT","NDSN","SEE","LNT",
    "ALLE","LW","HRL","BEN","AOS","FMC","JNPR","RJF","KDP","PKG",
    "PNR","MHK","WRK","NWSA","NWS","IVZ","CPT","TPR","LKQ","AIZ",
    "TECH","XRAY","MOS","IPG","NTNX","AAP","PARA","GNW","DXC","ROL"
]

# -----------------------
# Utility: fetch data with MAs
# -----------------------
def get_stock_data(ticker, period="6mo"):
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df.empty:
            return None
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["SMA200"] = df["Close"].rolling(200).mean()
        return df
    except:
        return None

# -----------------------
# Strategy conditions
# -----------------------
def check_momentum(df, crossover_days=7):
    if df is None or len(df) < 50:
        return False
    price = df["Close"].iloc[-1]
    sma20, sma50 = df["SMA20"].iloc[-1], df["SMA50"].iloc[-1]
    cond_price = price > sma20 and sma20 > sma50
    cross = (df["SMA20"].iloc[-crossover_days:] > df["SMA50"].iloc[-crossover_days:]).any()
    return cond_price and cross

def check_breakout(df):
    if df is None or len(df) < 50:
        return False
    price = df["Close"].iloc[-1]
    sma20, sma50 = df["SMA20"].iloc[-1], df["SMA50"].iloc[-1]
    cond_down = price < sma20 and sma20 > sma50
    cond_consolidate = df["Close"].iloc[-5:].max() - df["Close"].iloc[-5:].min() < 0.05 * price
    return cond_down and cond_consolidate

def check_pullback(df):
    if df is None or len(df) < 200:
        return False
    price = df["Close"].iloc[-1]
    sma20, sma50, sma200 = df["SMA20"].iloc[-1], df["SMA50"].iloc[-1], df["SMA200"].iloc[-1]
    cond_downtrend = price < sma20 and sma20 < sma50 and sma50 < sma200
    cond_recovery = price > sma20 and sma20 > sma50 and price < sma200
    return cond_recovery and not cond_downtrend

# -----------------------
# Screening engine
# -----------------------
def screen_stocks(tickers, strategy, crossover_days=7):
    results = []
    for t in tickers:
        df = get_stock_data(t)
        if df is None:
            continue
        if strategy == "Momentum" and check_momentum(df, crossover_days):
            results.append(t)
        elif strategy == "Breakout" and check_breakout(df):
            results.append(t)
        elif strategy == "Pullback" and check_pullback(df):
            results.append(t)
    return results

# -----------------------
# Streamlit UI
# -----------------------
st.title("📈 S&P 500 Multi-Style Stock Screener")

strategy = st.sidebar.selectbox("Select Strategy", ["Momentum", "Breakout", "Pullback"])
crossover_days = st.sidebar.slider("Momentum: Max crossover days", 3, 14, 7)

if st.button("Run Screener"):
    with st.spinner("Screening stocks..."):
        matches = screen_stocks(SP500_TICKERS, strategy, crossover_days)
    if matches:
        st.success(f"✅ Found {len(matches)} matching stocks for {strategy}:")
        st.dataframe(pd.DataFrame(matches, columns=["Ticker"]))
    else:
        st.warning("No matches found. Try relaxing conditions or extending period.")



