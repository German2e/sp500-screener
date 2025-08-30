import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List

# -----------------------------
# Get S&P500 tickers (fixed list)
# -----------------------------
@st.cache_data
def get_sp500_tickers() -> List[str]:
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
# Condition checkers (NaN safe + partial match)
# -----------------------------
def check_conditions(df: pd.DataFrame, style: str, params: dict) -> dict:
    df = df.copy()
    df["SMA20"] = sma(df["Close"], params.get("sma20", 20))
    df["SMA50"] = sma(df["Close"], params.get("sma50", 50))
    df["SMA200"] = sma(df["Close"], params.get("sma200", 200))
    df["RSI14"] = rsi(df["Close"], 14)
    df["VOL20"] = sma(df["Volume"], 20)

    latest = df.iloc[-1]
    partial = {}

    if style == "Momentum + Breakout":
        partial["SMA20>SMA50"] = pd.notna(latest["SMA20"]) and pd.notna(latest["SMA50"]) and latest["SMA20"] > latest["SMA50"]
        partial["RSI_in_range"] = pd.notna(latest["RSI14"]) and 40 <= latest["RSI14"] <= 60
        partial["Vol>Vol20"] = pd.notna(latest["VOL20"]) and latest["Volume"] > latest["VOL20"]
        consolidation_high = df["Close"].iloc[-(params["lookback_days"]+1):-1].max()
        partial["Breakout"] = pd.notna(consolidation_high) and latest["Close"] > consolidation_high and latest["Close"] <= consolidation_high * (1 + params["breakout_buffer"])
        meets = all(partial.values())

    elif style == "Pullback":
        partial["Close<SMA20"] = pd.notna(latest["SMA20"]) and latest["Close"] < latest["SMA20"]
        partial["Close>SMA50"] = pd.notna(latest["SMA50"]) and latest["Close"] > latest["SMA50"]
        partial["RSI<50"] = pd.notna(latest["RSI14"]) and latest["RSI14"] < 50
        meets = all(partial.values())

    elif style == "MA Crossover":
        partial["SMA20>SMA50>SMA200"] = pd.notna(latest["SMA20"]) and pd.notna(latest["SMA50"]) and pd.notna(latest["SMA200"]) and latest["SMA20"] > latest["SMA50"] > latest["SMA200"]
        meets = all(partial.values())

    elif style == "RSI Range":
        partial["RSI_in_range"] = pd.notna(latest["RSI14"]) and 40 <= latest["RSI14"] <= 60
        meets = all(partial.values())

    else:
        meets = False

    return {"meets": meets, "partial": partial}

# -----------------------------
# Screen stocks
# -----------------------------
def screen_stocks(tickers: List[str], style: str, **params) -> pd.DataFrame:
    results = []
    for t in tickers:
        try:
            df = fetch_data(t)
            conds = check_conditions(df, style, params)
            latest_close = df["Close"].iloc[-1]
            latest_rsi = rsi(df["Close"]).iloc[-1] if not df.empty else None
            sma20_val = sma(df["Close"], 20).iloc[-1] if not df.empty else None
            sma50_val = sma(df["Close"], 50).iloc[-1] if not df.empty else None
            vol = df["Volume"].iloc[-1] if not df.empty else None
            vol20 = sma(df["Volume"], 20).iloc[-1] if not df.empty else None

            results.append({
                "Ticker": t,
                "Meets_Entry": conds["meets"],
                "Partial_Match": conds["partial"],
                "Close": round(latest_close,2) if latest_close else None,
                "RSI14": round(latest_rsi,2) if latest_rsi else None,
                "SMA20": round(sma20_val,2) if sma20_val else None,
                "SMA50": round(sma50_val,2) if sma50_val else None,
                "Volume": int(vol) if vol else None,
                "VOL20": int(vol20) if vol20 else None
            })
        except Exception as e:
            print(f"Skipping {t}: {e}")
    return pd.DataFrame(results)

# -----------------------------
# Streamlit interface
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
    st.info(f"Scanning {len(tickers)} tickers for {style} strategy...")
    df_results = screen_stocks(tickers, style, **params)

    if not df_results.empty:
        st.subheader("Sample of scanned tickers")
        st.dataframe(df_results.head(10))

        # Show partial match info
        st.subheader("Partial match examples")
        st.dataframe(df_results[["Ticker","Partial_Match"]].head(10))

        matches = df_results[df_results["Meets_Entry"]]
        st.success(f"Found {len(matches)} full matches!")
        if not matches.empty:
            st.dataframe(matches)
            csv = matches.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Matches as CSV",
                data=csv,
                file_name=f"{style.replace(' ','_')}_matches.csv",
                mime="text/csv"
            )
    else:
        st.warning("No data returned — check fetch_data or parameters.")
