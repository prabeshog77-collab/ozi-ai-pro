# ozi-ai-pro
OZI AI PRO - Market Analysis Dashboard
# QUOTEX AI PRO — SINGLE-FILE MARKET ANALYSIS APP
# Educational analysis only. Not connected to Quotex and does not execute trades.
# Install dependencies once:
#   python -m pip install streamlit pandas yfinance plotly
# Run:
#   python -m streamlit run quotex_ai_pro_one_file.py
#
# Optional password (Windows Command Prompt):
#   set APP_PASSWORD=your-strong-password
# Optional password (macOS/Linux):
#   export APP_PASSWORD='your-strong-password'

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="QUOTEX AI PRO | Market Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- ACCESS GATE -------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD", "").strip()
if APP_PASSWORD and not st.session_state.get("authenticated", False):
    st.markdown("# 🔐 QUOTEX AI PRO")
    st.caption("Private local market-analysis dashboard")
    supplied = st.text_input("Password", type="password")
    if st.button("Unlock dashboard", use_container_width=True):
        if supplied == APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# ------------------------- STYLES -------------------------
st.markdown("""
<style>
    .stApp { background: #0b1020; color: #e5e7eb; }
    [data-testid="stSidebar"] { background: #111827; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #111827; border: 1px solid #263247;
        padding: 14px 16px; border-radius: 14px;
    }
    .hero {
        background: linear-gradient(120deg, #111827, #172554);
        border: 1px solid #263247; border-radius: 18px;
        padding: 20px 24px; margin-bottom: 16px;
    }
    .small-muted { color: #9ca3af; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1 style="margin:0">📈 QUOTEX AI PRO</h1>
  <p style="margin:8px 0 0 0;color:#cbd5e1">
  Market scanner · Technical analysis · Candlestick charts · Educational signals
  </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Educational tool only. This app does not connect to Quotex, execute trades, "
    "or access Quotex OTC prices. Market data may be delayed or unavailable. "
    "Indicator alignment is not a guaranteed prediction or financial advice."
)

# ------------------------- MARKET CONFIG -------------------------
INSTRUMENTS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "Gold futures": "GC=F",
    "Crude oil futures": "CL=F",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}
INTERVALS = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "1 day": "1d",
}
PERIODS = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]

with st.sidebar:
    st.header("⚙️ Settings")
    selected_names = st.multiselect(
        "Pairs to scan",
        list(INSTRUMENTS.keys()),
        default=["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "Gold futures"],
    )
    chosen_name = st.selectbox("Chart instrument", list(INSTRUMENTS.keys()), index=0)
    interval_label = st.selectbox("Candle interval", list(INTERVALS.keys()), index=2)
    period = st.selectbox("History period", PERIODS, index=2)
    min_history = st.slider("Minimum candles required", 30, 200, 50, 10)
    run_scan = st.button("🔄 Refresh / Scan markets", use_container_width=True)
    st.caption("Data source: Yahoo Finance. Availability and delays vary by symbol and interval.")

interval = INTERVALS[interval_label]

# ------------------------- DATA -------------------------
@st.cache_data(ttl=60, show_spinner=False)
def download_data(ticker: str, selected_period: str, selected_interval: str, refresh_token: int):
    try:
        frame = yf.download(
            ticker,
            period=selected_period,
            interval=selected_interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if frame is None or frame.empty:
            return pd.DataFrame()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(c).title() for c in frame.columns]
        required = ["Open", "High", "Low", "Close"]
        if not all(col in frame.columns for col in required):
            return pd.DataFrame()
        frame = frame.dropna(subset=required).copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame
    except Exception:
        return pd.DataFrame()

def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["Close"]
    out["EMA 9"] = close.ewm(span=9, adjust=False).mean()
    out["EMA 21"] = close.ewm(span=21, adjust=False).mean()
    out["EMA 50"] = close.ewm(span=50, adjust=False).mean()
    out["EMA 200"] = close.ewm(span=200, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, float("nan"))
    out["RSI 14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD histogram"] = out["MACD"] - out["MACD signal"]

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["BB middle"] = middle
    out["BB upper"] = middle + 2 * std
    out["BB lower"] = middle - 2 * std

    high_low = out["High"] - out["Low"]
    high_close = (out["High"] - close.shift()).abs()
    low_close = (out["Low"] - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    out["ATR 14"] = true_range.rolling(14).mean()
    return out

def summarize(frame: pd.DataFrame, minimum: int):
    if frame.empty:
        return {"status": "NO DATA", "direction": "UNAVAILABLE", "reason": "No market data returned."}
    data = add_indicators(frame)
    last = data.iloc[-1]
    if len(data) < minimum or pd.isna(last["RSI 14"]) or pd.isna(last["MACD signal"]):
        return {
            "status": "INSUFFICIENT DATA", "direction": "NO TRADE",
            "reason": f"Need more valid candles (at least {minimum}).",
            "data": data, "last": last
        }

    up_trend = last["EMA 9"] > last["EMA 21"] > last["EMA 50"]
    down_trend = last["EMA 9"] < last["EMA 21"] < last["EMA 50"]
    up_momentum = last["MACD"] > last["MACD signal"] and last["RSI 14"] > 50
    down_momentum = last["MACD"] < last["MACD signal"] and last["RSI 14"] < 50

    if up_trend and up_momentum:
        direction = "BULLISH CONDITIONS"
        reason = "EMA 9 > EMA 21 > EMA 50, MACD above its signal, and RSI above 50."
    elif down_trend and down_momentum:
        direction = "BEARISH CONDITIONS"
        reason = "EMA 9 < EMA 21 < EMA 50, MACD below its signal, and RSI below 50."
    else:
        direction = "MIXED — NO CLEAR SETUP"
        reason = "Trend and momentum indicators do not agree."
    return {"status": "OK", "direction": direction, "reason": reason, "data": data, "last": last}

refresh_token = int(datetime.now().timestamp()) if run_scan else 0

# ------------------------- MARKET SCANNER -------------------------
st.subheader("🌐 Market Scanner")
if not selected_names:
    st.info("Select one or more instruments in the sidebar.")
else:
    scan_rows = []
    with st.spinner("Scanning selected instruments…"):
        for name in selected_names:
            frame = download_data(INSTRUMENTS[name], period, interval, refresh_token)
            result = summarize(frame, min_history)
            if result.get("status") == "OK":
                last = result["last"]
                rsi_value = last["RSI 14"]
                scan_rows.append({
                    "Instrument": name,
                    "Latest close": float(last["Close"]),
                    "RSI (14)": round(float(rsi_value), 2) if pd.notna(rsi_value) else None,
                    "EMA trend": "UP" if last["EMA 9"] > last["EMA 21"] > last["EMA 50"] else (
                        "DOWN" if last["EMA 9"] < last["EMA 21"] < last["EMA 50"] else "MIXED"
                    ),
                    "Analysis": result["direction"],
                    "Latest candle (UTC)": frame.index[-1].strftime("%Y-%m-%d %H:%M"),
                    "Candles": len(frame),
                })
            else:
                scan_rows.append({
                    "Instrument": name, "Latest close": None, "RSI (14)": None,
                    "EMA trend": "UNKNOWN", "Analysis": result["status"],
                    "Latest candle (UTC)": "—", "Candles": len(frame),
                })
    st.dataframe(pd.DataFrame(scan_rows), use_container_width=True, hide_index=True)

# ------------------------- SELECTED CHART -------------------------
st.subheader(f"📊 Detailed Analysis — {chosen_name}")
chart_frame = download_data(INSTRUMENTS[chosen_name], period, interval, refresh_token)
chart_result = summarize(chart_frame, min_history)

if chart_frame.empty:
    st.error("No data returned. Try another instrument, interval, or history period.")
else:
    analyzed = add_indicators(chart_frame)
    latest = analyzed.iloc[-1]
    a, b, c, d = st.columns(4)
    a.metric("Latest close", f"{latest['Close']:,.5f}")
    b.metric("RSI (14)", "—" if pd.isna(latest["RSI 14"]) else f"{latest['RSI 14']:.2f}")
    c.metric("MACD", "—" if pd.isna(latest["MACD"]) else f"{latest['MACD']:.5f}")
    d.metric("Analysis", chart_result["direction"])

    st.write("**Reasoning:**", chart_result["reason"])
    st.caption(
        f"Latest candle: {analyzed.index[-1].strftime('%Y-%m-%d %H:%M UTC')} · "
        f"Retrieved from Yahoo Finance · Historical/streaming availability varies."
    )

    tabs = st.tabs(["Candlestick chart", "RSI", "MACD", "Data table", "Method & limits"])
    with tabs[0]:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=analyzed.index, open=analyzed["Open"], high=analyzed["High"],
            low=analyzed["Low"], close=analyzed["Close"], name="Price"
        ))
        for col in ["EMA 9", "EMA 21", "EMA 50"]:
            fig.add_trace(go.Scatter(x=analyzed.index, y=analyzed[col], name=col, mode="lines"))
        fig.update_layout(
            template="plotly_dark", height=600, xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h")
        )
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=analyzed.index, y=analyzed["RSI 14"], name="RSI 14"))
        fig_rsi.add_hline(y=70, line_dash="dash")
        fig_rsi.add_hline(y=30, line_dash="dash")
        fig_rsi.update_layout(template="plotly_dark", height=320, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    with tabs[2]:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=analyzed.index, y=analyzed["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=analyzed.index, y=analyzed["MACD signal"], name="Signal"))
        fig_macd.add_trace(go.Bar(x=analyzed.index, y=analyzed["MACD histogram"], name="Histogram"))
        fig_macd.update_layout(template="plotly_dark", height=320)
        st.plotly_chart(fig_macd, use_container_width=True)
    with tabs[3]:
        columns = [c for c in [
            "Open", "High", "Low", "Close", "Volume", "EMA 9", "EMA 21",
            "EMA 50", "RSI 14", "MACD", "MACD signal", "ATR 14"
        ] if c in analyzed.columns]
        st.dataframe(analyzed[columns].tail(200), use_container_width=True)
        csv = analyzed[columns].to_csv().encode("utf-8")
        st.download_button("Download indicator data as CSV", csv, "market_analysis.csv", "text/csv")
    with tabs[4]:
        st.markdown("""
        **How the conditions are calculated**
        - Bullish conditions: EMA 9 > EMA 21 > EMA 50, MACD above its signal, RSI > 50.
        - Bearish conditions: EMA 9 < EMA 21 < EMA 50, MACD below its signal, RSI < 50.
        - Otherwise: mixed conditions / no clear setup.

        These are rule-based indicator summaries, not AI certainty or calibrated probabilities.
        They do not account for Quotex execution prices, payout rules, slippage, or OTC-specific pricing.
        """)
        st.write("App timestamp (UTC):", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))

st.divider()
st.caption("QUOTEX AI PRO · Educational market analysis only · No automated trading · No guaranteed accuracy")
