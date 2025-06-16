import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# streamlit_app.py



"""
Streamlit app: Real‑time stock analysis with technical & fundamental metrics
Dependencies: streamlit, yfinance, pandas, numpy, pandas_ta (optional)
Run with: streamlit run streamlit_app.py
"""

import time
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def fetch_price_history(ticker: str, period: str = "1mo", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("No price data returned; check ticker or interval")
        return df
    except Exception as e:
        st.error(f"Error fetching price data: {e}")
        return pd.DataFrame()


def _clean_value(value: Any) -> Any:
    return value if value is not None else "N/A"


def fetch_fundamentals(ticker: str) -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)
        raw: Dict[str, Any] = stock.info or {}
        keys = [
            "sector",
            "industry",
            "marketCap",
            "beta",
            "trailingPE",
            "forwardPE",
            "trailingEps",
            "dividendYield",
            "profitMargins",
        ]
        return {k: _clean_value(raw.get(k)) for k in keys}
    except Exception as e:
        st.warning(f"Could not retrieve fundamentals: {e}")
        return {}


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = np.where(roll_down == 0, np.nan, roll_up / roll_down)
    df["RSI14"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def analyze(df: pd.DataFrame, fundamentals: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    if df.empty:
        return {"error": "No data"}

    last = df["Close"].iloc[-1]
    rsi = df["RSI14"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    signal_val = df["Signal"].iloc[-1]

    support = df["Close"].tail(20).min()
    resistance = df["Close"].tail(20).max()

    returns = df["Close"].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)

    mean_daily = returns.tail(7).mean()
    roi = round(mean_daily * 5 * 100, 2)

    if vol > 0.5:
        risk = "muy alto"
    elif vol > 0.3:
        risk = "alto"
    elif vol > 0.2:
        risk = "moderado"
    else:
        risk = "bajo"

    entry = round(last * 0.985, 2)
    exit_p = round(last * 1.05, 2)

    return {
        "ticker": ticker.upper(),
        "precio_actual": round(last, 2),
        "soporte": round(support, 2),
        "resistencia": round(resistance, 2),
        "RSI14": round(rsi, 2),
        "MACD": round(macd, 4),
        "Signal": round(signal_val, 4),
        "volatilidad_anual": round(vol, 2),
        "riesgo": risk,
        "roi_estimado_semana_%": roi,
        "entrada_sugerida": entry,
        "salida_objetivo": exit_p,
        "fundamentals": fundamentals,
    }


st.set_page_config(page_title="Análisis de Acción", layout="centered")
st.title("📈 Análisis Técnico y Fundamental")

ticker = st.sidebar.text_input("Ticker", "NVDA").upper()
period = st.sidebar.selectbox("Periodo", ["7d", "1mo", "3mo", "6mo", "1y"], index=1)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1h", "30m"], index=0)

if st.sidebar.button("Analizar"):
    df_prices = fetch_price_history(ticker, period, interval)
    df_indicators = compute_indicators(df_prices)
    fundamentals_data = fetch_fundamentals(ticker)
    analysis = analyze(df_indicators, fundamentals_data, ticker)

    st.json(analysis)

    if not df_indicators.empty:
        st.line_chart(df_indicators[["Close", "SMA20", "SMA50"]].dropna(), height=300)
        st.line_chart(df_indicators[["RSI14"]].dropna(), height=200)
        st.line_chart(df_indicators[["MACD", "Signal"]].dropna(), height=200)
