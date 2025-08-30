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
# Fetch data
# -----------------------------
def fetch_data(ticker: str, period: str = "240d", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    return df

# -----------------------------
# Condition checkers for each style (NaN-safe, partial match tracking)
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

    # Momentum + Breakout
    if style == "Momentum + Breakout":
        cond_ma = latest["SMA20"] > latest["SMA50"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) else False
        cond_rsi = 40 <= latest["RSI14"] <= 60 if not pd.isna(latest["RSI14"]) else False
        cond_vol = latest["Volume"] > latest["VOL20"] if not pd.isna(latest["VOL20"]) else False
        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        cond_breakout = (latest["Close"] > consolidation_high) and (latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"])) if not pd.isna(consolidation_high) else False

        partial = {
            "SMA20>SMA50": cond_ma,
            "RSI40-60": cond_rsi,
            "Volume>VOL20": cond_vol,
            "Breakout": cond_breakout
        }
        return cond_ma and cond_rsi and cond_vol and cond_breakout, partial

    # Pullback
    elif style == "Pullback":
        cond_pullback = latest["Close"] < latest["SMA20"] and latest["Close"] > latest["SMA50"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) else False
        cond_rsi = latest["RSI14"] < 50 if not pd.isna(latest["RSI14"]) else False
        partial = {"Pullback": cond_pullback, "RSI<50": cond_rsi}
        return cond_pullback and cond_rsi, partial

    # MA Crossover
    elif style == "MA Crossover":
        cond_ma = latest["SMA20"] > latest["SMA50"] and latest["SMA50"] > latest["SMA200"] if not pd.isna(latest["SMA20"]) and not pd.isna(latest["SMA50"]) and not pd.isna(latest["SMA200"]) else False
        partial = {"MA Crossover": cond_ma}
        return cond_ma, partial

    # RSI Range
    elif style == "RSI Range":
        cond_rsi = 40 <= latest["RSI14"] <= 60 if not pd.isna(latest["RSI14"]) else False
        partial = {"RSI40-60": cond_rsi}
        return cond_rsi, partial

    return False, {}

import time  # needed for retry delays

# -----------------------------
# Main screener with fetch retry
# -----------------------------
def screen_stocks(tickers: List[str], style: str, retries: int = 3, delay: float = 1.0, **params) -> pd.DataFrame:
    results = []
    for t in tickers:
        df = None
        for attempt in range(retries):
            try:
                df = fetch_data(t)
                break  # success
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {t}: {e}")
                time.sleep(delay)
        if df is None or df.empty:
            print(f"Skipping {t}: No data after {retries} attempts")
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

    return pd.DataFrame(results)
