import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from typing import List
import time

# -----------------------------
# Preloaded S&P 500 tickers (latest)
# -----------------------------
import streamlit as st
from typing import List

@st.cache_data
def get_sp500_tickers() -> List[str]:
    tickers = [
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
# Fetch data safely
# -----------------------------
def fetch_data(ticker: str, period: str = "240d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return None
    return df

# -----------------------------
# Condition checkers
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict):
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

    try:
        # Momentum + Breakout
        if style == "Momentum + Breakout":
            cond_ma = latest["SMA20"] > latest["SMA50"]
            cond_rsi = 40 <= latest["RSI14"] <= 60
            cond_vol = latest["Volume"] > latest["VOL20"]
            consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
            cond_breakout = (latest["Close"] > consolidation_high) and (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"]))
            partial = {
                "SMA20>SMA50": cond_ma,
                "RSI40-60": cond_rsi,
                "Volume>VOL20": cond_vol,
                "Breakout": cond_breakout
            }
            return all(partial.values()), partial

        # Pullback
        elif style == "Pullback":
            cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"]
            cond_rsi = latest["RSI14"] < 50
            partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
            return all(partial.values()), partial

        # MA Crossover
        elif style == "MA Crossover":
            cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"]
            partial = {"MA Crossover": cond_ma}
            return cond_ma, partial

        # RSI Range
        elif style == "RSI Range":
            cond_rsi = 40 <= latest["RSI14"] <= 60
            partial = {"RSI40-60": cond_rsi}
            return cond_rsi, partial

    except Exception as e:
        return False, {}

    return False, {}

# -----------------------------
# Screener
# -----------------------------
def screen_stocks(tickers: List[str], style: str, retries: int = 2, delay: float = 0.5, **params):
    results = []
    progress = st.progress(0)
    total = len(tickers)

    for i, t in enumerate(tickers):
        df = None
        for attempt in range(retries):
            df = fetch_data(t)
            if df is not None:
                break
            time.sleep(delay)

        if df is None or df.empty:
            continue

        meets, partial = check_conditions(df, style, params)

        latest_close = df["Close"].iloc[-1] if not df.empty else None
        latest_rsi = rsi(df["Close"]).iloc[-1] if not df.empty else None
        sma20_val = sma(df["Close"], 20).iloc[-1] if not df.empty else None
        sma50_val = sma(df["Close"], 50).iloc[-1] if not df.empty else None
        vol = df["Volume"].iloc[-1] if not df.empty else None
        vol20 = sma(df["Volume"], 20).iloc[-1] if not df.empty else None

        results.append({
            "Ticker": t,
            "Meets_Entry": meets,
            "Close": round(latest_close, 2) if latest_close is not None else None,
            "RSI14": round(latest_rsi, 2) if latest_rsi is not None else None,
            "SMA20": round(sma20_val, 2) if sma20_val is not None else None,
            "SMA50": round(sma50_val, 2) if sma50_val is not None else None,
            "Volume": int(vol) if vol is not None else None,
            "VOL20": int(vol20) if vol20 is not None else None,
            "Partial_Match": partial
        })
        progress.progress((i+1)/total)

    return pd.DataFrame(results)

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("Multi-Style Stock Screener (S&P500)")

style = st.sidebar.selectbox("Select Trading Style:", ["Momentum + Breakout", "Pullback", "MA Crossover", "RSI Range"])

params = {}
params["lookback_days"] = st.sidebar.slider("Lookback Days (Momentum/Breakout)", 10, 60, 30)
params["breakout_buffer"] = st.sidebar.slider("Breakout Buffer (%)", 0.01, 0.10, 0.05)
params["sma20"] = st.sidebar.slider("SMA20", 10, 50, 20)
params["sma50"] = st.sidebar.slider("SMA50", 20, 100, 50)
params["sma200"] = st.sidebar.slider("SMA200", 50, 250, 200)

if st.sidebar.button("Run Screener"):
    tickers = get_sp500_tickers()
    if not tickers:
        st.warning("No tickers to scan.")
    else:
        st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")
        df_results = screen_stocks(tickers, style, **params)
        if "Meets_Entry" not in df_results.columns:
            st.warning("No valid stock data returned. Try again later or adjust filters.")
        else:
            matches = df_results[df_results["Meets_Entry"] == True]
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


