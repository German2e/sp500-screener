import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers (fixed list)
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> list[str]:
    tickers = [
        "MMM","ACE","ABT","ANF","ACN","ADBE","AMD","AES","AET","AFL","A","GAS","APD",
        "ARG","AKAM","AA","ALXN","ATI","AGN","ALL","ANR","ALTR","MO","AMZN","AEE","AEP",
        "AXP","AIG","AMT","AMP","ABC","AMGN","APH","APC","ADI","AON","APA","AIV","APOL",
        "AAPL","AMAT","ADM","AIZ","T","ADSK","ADP","AN","AZO","AVB","AVY","AVP","BHI",
        "BLL","BAC","BK","BCR","BAX","BBT","BEAM","BDX","BBBY","BMS","BRK.B","BBY","BIG",
        "BIIB","BLK","HRB","BMC","BA","BWA","BXP","BSX","BMY","BRCM","BF.B","CHRW","CA",
        "CVC","COG","CAM","CPB","COF","CAH","CFN","KMX","CCL","CAT","CBG","CBS","CELG",
        "CNP","CTL","CERN","CF","SCHW","CHK","CVX","CMG","CB","CI","CINF","CTAS","CSCO",
        "C","CTXS","CLF","CLX","CME","CMS","COH","KO","CCE","CTSH","CL","CMCSA","CMA",
        "CSC","CAG","COP","CNX","ED","STZ","CBE","GLW","COST","CVH","COV","CCI","CSX",
        "CMI","CVS","DHI","DHR","DRI","DVA","DF","DE","DELL","DNR","XRAY","DVN","DV","DO",
        "DTV","DFS","DISCA","DLTR","D","RRD","DOV","DOW","DPS","DTE","DD","DUK","DNB",
        "ETFC","EMN","ETN","EBAY","ECL","EIX","EW","EA","EMC","EMR","ESV","ETR","EOG",
        "EQT","EFX","EQR","EL","EXC","EXPE","EXPD","ESRX","XOM","FFIV","FDO","FAST","FII",
        "FDX","FIS","FITB","FHN","FSLR","FE","FISV","FLIR","FLS","FLR","FMC","FTI","F",
        "FRX","FOSL","BEN","FCX","FTR","GME","GCI","GPS","GD","GE","GIS","GPC","GNW","GILD",
        "GS","GT","GOOG","GWW","HAL","HOG","HAR","HRS","HIG","HAS","HCP","HCN","HNZ","HP",
        "HES","HPQ","HD","HON","HRL","HSP","HST","HCBK","HUM","HBAN","ITW","IR","TEG","INTC",
        "ICE","IBM","IFF","IGT","IP","IPG","INTU","ISRG","IVZ","IRM","JBL","JEC","JDSU",
        "JNJ","JCI","JOY","JPM","JNPR","K","KEY","KMB","KIM","KMI","KLAC","KSS","KFT","KR",
        "LLL","LH","LRCX","LM","LEG","LEN","LUK","LXK","LIFE","LLY","LTD","LNC","LLTC","LMT",
        "L","LO","LOW","LSI","MTB","M","MRO","MPC","MAR","MMC","MAS","MA","MAT","MKC","MCD",
        "MHP","MCK","MJN","MWV","MDT","MRK","MET","PCS","MCHP","MU","MSFT","MOLX","TAP","MON",
        "MNST","MCO","MS","MOS","MSI","MUR","MYL","NBR","NDAQ","NOV","NTAP","NFLX","NWL","NFX",
        "NEM","NWSA","NEE","NKE","NI","NE","NBL","JWN","NSC","NTRS","NOC","NU","NRG","NUE",
        "NVDA","NYX","ORLY","OXY","OMC","OKE","ORCL","OI","PCAR","PLL","PH","PDCO","PAYX",
        "BTU","JCP","PBCT","POM","PEP","PKI","PRGO","PFE","PCG","PM","PSX","PNW","PXD","PBI",
        "PCL","PNC","RL","PPG","PPL","PX","PCP","PCLN","PFG","PG","PGR","PLD","PRU","PEG",
        "PSA","PHM","QEP","PWR","QCOM","DGX","RRC","RTN","RHT","RF","RSG","RAI","RHI","ROK",
        "COL","ROP","ROST","RDC","R","SWY","SAI","CRM","SNDK","SCG","SLB","SNI","STX","SEE",
        "SHLD","SRE","SHW","SIAL","SPG","SLM","SJM","SNA","SO","LUV","SWN","SE","S","STJ",
        "SWK","SPLS","SBUX","HOT","STT","SRCL","SYK","SUN","STI","SYMC","SYY","TROW","TGT",
        "TEL","TE","THC","TDC","TER","TSO","TXN","TXT","HSY","TRV","TMO","TIF","TWX","TWC",
        "TIE","TJX","TMK","TSS","TRIP","TSN","TYC","USB","UNP","UNH","UPS","X","UTX","UNM",
        "URBN","VFC","VLO","VAR","VTR","VRSN","VZ","VIAB","V","VNO","VMC","WMT","WAG","DIS",
        "WPO","WM","WAT","WPI","WLP","WFC","WDC","WU","WY","WHR","WFM","WMB","WIN","WEC",
        "WPX","WYN","WYNN","XEL","XRX","XLNX","XL","XYL","YHOO","YUM","ZMH","ZION"
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
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()  # Return empty DF instead of raising error
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

# -----------------------------
# Condition checkers
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> bool:
    if df.empty:
        return False

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
        cond_breakout = (latest["Close"] > consolidation_high) and \
                        (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"]))
        return bool(cond_ma and cond_rsi and cond_vol and cond_breakout)

    elif style == "Pullback":
        cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"]
        cond_rsi = latest["RSI14"] < 50
        return bool(cond_pullback and cond_rsi)

    elif style == "MA Crossover":
        cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"]
        return bool(cond_ma)

    elif style == "RSI Range":
        return 40 <= latest["RSI14"] <= 

