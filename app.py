import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List

# -----------------------------
# Preloaded S&P 500 tickers
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> List[str]:
    return [
        "MMM","AOS","ABT","ABBV","ACN","ATVI","ADBE","AMD","AAP","AES","AFL","A","APD",
        "AKAM","ALK","ALB","ARE","ALXN","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
        "AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","ABC","AME","AMGN","APH","ADI",
        "ANSS","ANTM","AON","AOS","APA","AAPL","AMAT","APTV","ADM","ARNC","ANET","AJG","AIZ",
        "T","ADSK","ADP","AZO","AVB","AVY","BKR","BLL","BAC","BK","BAX","BDX","BRK.B","BBY",
        "BIO","BIIB","BLK","BA","BKNG","BWA","BXP","BSX","BMY","AVGO","BR","BF.B","CHRW","COG",
        "CDNS","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","CNC",
        "CNP","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO",
        "C","CFG","CTXS","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ",
        "COO","CPRT","GLW","CTVA","COST","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE",
        "DAL","XRAY","DVN","DXC","FANG","DLR","DFS","DISCA","DISCK","DISH","DG","DLTR","D",
        "DOV","DOW","DTE","DUK","DRE","DD","DXC","ETFC","EMN","ETN","EBAY","ECL","EIX","EW","EA",
        "EMR","ETR","EVRG","ESS","EL","ETSY","RE","EXC","EXPE","EXPD","EXR","XOM","FFIV","FB",
        "FAST","FRT","FDX","FIS","FITB","FRC","FE","FISV","FLT","FMC","F","FTNT","FTV","FBHS",
        "FOXA","FOX","BEN","FCX","GPS","GRMN","IT","GE","GNRC","GD","GEHC","GILD","GL","GLPI",
        "GM","GOOG","GOOGL","GPC","GILD","GIS","GM","GPN","GS","GWW","HAL","HBI","HIG","HAS",
        "HCA","PEAK","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ",
        "HUM","HBAN","HII","IEX","IDXX","INFO","ITW","ILMN","INCY","IR","INTC","ICE","IBM","IP",
        "IPG","IFF","INTU","ISRG","IVZ","IPGP","IQV","IRM","JKHY","J","JBHT","SJM","JNJ","JCI",
        "JPM","JNPR","KSU","K","KEY","KEYS","KMB","KIM","KMI","KLAC","KSS","KHC","KR","LHX",
        "LH","LRCX","LM","LEG","LEN","LLY","LNC","LIN","LYV","LKQ","LMT","L","LOW","LUMN","LYB",
        "MTB","MAC","M","MRO","MPC","MAR","MMC","MAS","MA","MKTX","MAT","MKC","MCD","MCK","MDT",
        "MRK","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MHK","TAP","MDLZ","MPWR","MNST","MCO",
        "MS","MOS","MSI","MSCI","MYL","NDAQ","NOV","NKTR","NTAP","NFLX","NWL","NEM","NEE","NLSN",
        "NKE","NI","NBL","JWN","NSC","NTRS","NOC","NUE","NVDA","NVR","NXPI","ORLY","OXY","OMC",
        "OKE","ORCL","OTIS","PCAR","PKG","PARA","PAYX","PYPL","PNR","PBCT","PEP","PKI","PRGO",
        "PFE","PM","PSX","PNW","PXD","PNC","PPG","PPL","PFG","PG","PGR","PLD","PRU","PEG",
        "PSA","PH","PVH","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG",
        "RMD","RHI","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE",
        "NOW","SHW","SPG","SWKS","SNA","SO","LUV","SPGI","SWK","SBUX","STT","STE","SYK","SIVB",
        "SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TGT","TEL","TDY","TFX","TXN","TXT","TMO",
        "TJX","TSCO","TDG","TRV","TRMB","TFC","TWTR","TSN","UDR","ULTA","USB","UAA","UNP","UAL",
        "UNH","UPS","URI","UHS","VLO","VTR","VRSN","VZ","VFC","VIAC","V","VNO","VMC","WRB","WAB",
        "WMT","WBA","WM","WAT","WEC","WFC","WELL","WST","WSM","WY","WHR","WMB","WLTW","WYNN",
        "XEL","XRX","XLNX","XYL","YUM","ZBRA","ZBH","ZION","ZTS"
        # ... continue adding all remaining tickers until 500
    ]

# -----------------------------
# Moving averages
# -----------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

# -----------------------------
# Fetch data
# -----------------------------
def fetch_data(ticker: str, period: str = "600d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    return df

# -----------------------------
# Trading styles conditions
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> bool:
    if df is None or df.shape[0] < 50:
        return False

    df = df.copy()
    df["SMA20"] = sma(df["Close"], 20)
    df["SMA50"] = sma(df["Close"], 50)
    df["SMA200"] = sma(df["Close"], 200)
    df["VOL20"] = sma(df["Volume"], 20)

    latest = df.iloc[-1]
    df_recent = df.iloc[-params.get("lookback_days", 7):]

    try:
        if style == "Momentum":
            # Price above 20MA, 20MA > 50MA, 20MA crossed above 50MA in last N days
            recent_cross = ((df_recent["SMA20"] > df_recent["SMA50"]) &
                            (df_recent["SMA20"].shift(1) <= df_recent["SMA50"].shift(1)))
            return bool(latest["Close"] > latest["SMA20"] and
                        latest["SMA20"] > latest["SMA50"] and
                        recent_cross.any())

        elif style == "Breakout":
            # Price below 20MA, 20MA moving toward 50MA, price consolidating
            price_range = df_recent["Close"].max() - df_recent["Close"].min()
            recent_cross_down = ((df_recent["SMA20"] < df_recent["SMA50"]) &
                                (df_recent["SMA20"].shift(1) >= df_recent["SMA50"].shift(1)))
            return bool(latest["Close"] < latest["SMA20"] and
                        recent_cross_down.any() and
                        price_range / latest["Close"] < params.get("consolidation_pct", 0.05))

        elif style == "Recovery":
            # Stock previously downtrend, now recovering above 20MA, 20MA crossed above 50MA, both under 200MA
            cond_downtrend = latest["SMA20"] < latest["SMA50"] and latest["SMA50"] < latest["SMA200"]
            cond_recovery = latest["Close"] > latest["SMA20"] and latest["SMA20"] > latest["SMA50"] and latest["SMA50"] < latest["SMA200"]
            vol_check = latest["Volume"] > latest["VOL20"]
            return bool(cond_downtrend and cond_recovery and vol_check)
    except Exception:
        return False

    return False

# -----------------------------
# Screener
# -----------------------------
def screen_stocks(tickers: List[str], style: str, **params) -> pd.DataFrame:
    results = []
    progress = st.progress(0)
    total = len(tickers)
    for i, t in enumerate(tickers):
        df = fetch_data(t)
        meets = check_conditions(df, style, params)
        if df is not None:
            latest_close = df["Close"].iloc[-1]
            sma20_val = df["SMA20"].iloc[-1]
            sma50_val = df["SMA50"].iloc[-1]
            results.append({
                "Ticker": t,
                "Meets_Entry": meets,
                "Close": round(latest_close, 2),
                "SMA20": round(sma20_val, 2) if not np.isnan(sma20_val) else None,
                "SMA50": round(sma50_val, 2) if not np.isnan(sma50_val) else None,
            })
        progress.progress((i+1)/total)
    return pd.DataFrame(results)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("S&P 500 Multi-Style Screener")

style = st.sidebar.selectbox("Select Trading Style:", ["Momentum", "Breakout", "Recovery"])
params = {}
params["lookback_days"] = st.sidebar.slider("Lookback Days for Crosses", 3, 14, 7)
params["consolidation_pct"] = st.sidebar.slider("Consolidation % (Breakout)", 0.01, 0.1, 0.05)

if st.sidebar.button("Run Screener"):
    tickers = get_sp500_tickers()
    st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")
    df_results = screen_stocks(tickers, style, **params)
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


