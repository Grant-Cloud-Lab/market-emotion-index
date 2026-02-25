import os
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from dateutil import tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Market Emotion Index", layout="wide")

st.markdown("""
<style>
/* Main page padding */
.block-container { 
  padding-top: 0.6rem;
  padding-bottom: 0.6rem;
}

/* Reduce whitespace between elements */
div[data-testid="stVerticalBlock"] > div { 
  gap: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

LOCAL_TZ = tz.gettz("Africa/Johannesburg")
from datetime import datetime

st.sidebar.caption(
    f"Last refresh: {datetime.now(LOCAL_TZ).strftime('%Y-%m-%d %H:%M:%S')}"
)
import time

if "run_count" not in st.session_state:
    st.session_state.run_count = 0
st.session_state.run_count += 1

st.sidebar.caption(f"Run #: {st.session_state.run_count} | tick: {time.time():.0f}")

REFRESH_SECONDS = 300  # 5 minutes
# Score regimes (for labeling)
REGIME_THRESHOLDS = {
    "stress_build": (-80, -40),
    "caution": (-40, -10),
    "transition": (-10, 10),
    "risk_on": (10, 40),
    "speculative_heat": (40, 10**9),}

# Direction lookback:
# If your data updates every 5 minutes, 12 bars ≈ 1 hour.
# If it's daily bars, set this to 1 later.
DIR_LOOKBACK_BARS = 1   

# Flat band to prevent noise-flips in direction logic (0.05%)
DIR_FLAT_BAND = 0.0005

# Volatility expansion settings for emotion score
VOL_WINDOW = 50       # rolling window size
VOL_LOOKBACK = 10     # compare last vs first of recent vol points
DEFAULT_HALF_LIFE_HOURS = 6

FINANCE_KEYWORDS = [
    "market", "stocks", "shares", "equities", "bond", "yield", "rates",
    "fed", "inflation", "cpi", "pce", "recession", "growth", "jobs",
    "nasdaq", "s&p", "dow", "dollar", "dxy", "treasury", "oil", "gold",
    "bitcoin", "crypto", "ethereum", "vix", "volatility", "tariff"
]

PANIC_KEYWORDS = [
    "crash", "panic", "sell-off", "selloff", "plunge", "slump", "fear",
    "turmoil", "bank run", "default", "bankrupt", "collapse",
    "contagion", "liquidation", "downgrade", "emergency"
]

HISTORY_PATH = os.path.join("data", "history.csv")
# ----------------------------
# Market Drivers (VIX, 10Y, DXY, Nasdaq, BTC)
# ----------------------------

@st.cache_data(ttl=300)
def fetch_fred_latest(series_id: str):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().sort_values("date")
    latest = float(df.iloc[-1]["value"])
    prev = float(df.iloc[-2]["value"]) if len(df) >= 2 else float("nan")
    date_latest = df.iloc[-1]["date"]
    return latest, prev, date_latest

@st.cache_data(ttl=300)
def fetch_stooq_daily_latest(symbol: str):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    df = pd.read_csv(url)
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    if "date" not in df.columns or "close" not in df.columns:
        raise ValueError(f"Unexpected columns for {symbol}: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna().sort_values("date")

    if len(df) < 2:
        raise ValueError(f"Not enough data returned for {symbol}")

    latest = float(df.iloc[-1]["close"])
    prev = float(df.iloc[-2]["close"])
    date_latest = df.iloc[-1]["date"]

    return latest, prev, date_latest
# -------------------------------
# Decision Engine helpers
# -------------------------------

def label_regime(score: float) -> str:
    # Uses the ranges from REGIME_THRESHOLDS (set in Step 0)
    if score <= REGIME_THRESHOLDS["stress_build"][1]:
        return "🔴 Stress Build"
    if REGIME_THRESHOLDS["caution"][0] < score <= REGIME_THRESHOLDS["caution"][1]:
        return "🟡 Caution / Defensive Tilt"
    if REGIME_THRESHOLDS["transition"][0] < score < REGIME_THRESHOLDS["transition"][1]:
        return "⚪ Transition"
    if REGIME_THRESHOLDS["risk_on"][0] <= score < REGIME_THRESHOLDS["risk_on"][1]:
        return "🟢 Risk On"
    return "🟣 Speculative Heat"


def pct_change_over_n(series: pd.Series, n: int) -> float:
    if series is None or len(series) <= n:
        return 0.0
    a = float(series.iloc[-1])
    b = float(series.iloc[-1 - n])
    if b == 0 or np.isnan(b) or np.isnan(a):
        return 0.0
    return (a / b) - 1.0


def direction(series: pd.Series, n: int, flat_band: float = DIR_FLAT_BAND) -> str:
    r = pct_change_over_n(series, n)
    if r > flat_band:
        return "up"
    if r < -flat_band:
        return "down"
    return "flat"


def divergence_alert(nq_series: pd.Series, btc_series: pd.Series, vix_series: pd.Series, n: int) -> tuple[bool, str]:
    nq_ret = pct_change_over_n(nq_series, n)
    btc_ret = pct_change_over_n(btc_series, n)
    vix_ret = pct_change_over_n(vix_series, n)

    is_div = (nq_ret > 0) and (btc_ret < 0) and (vix_ret > 0)
    if is_div:
        return True, "⚠️ Internal Risk Divergence (NQ↑ BTC↓ VIX↑)"
    return False, "✅ No internal divergence (NQ/BTC/VIX)"


def macro_pulse(dxy_series: pd.Series, us10y_series: pd.Series, vix_series: pd.Series, n: int) -> tuple[str, dict]:
    inputs = {
        "DXY": direction(dxy_series, n) if dxy_series is not None else "missing",
        "10Y": direction(us10y_series, n) if us10y_series is not None else "missing",
        "VIX": direction(vix_series, n) if vix_series is not None else "missing",
    }

    ups = sum(1 for v in inputs.values() if v == "up")
    downs = sum(1 for v in inputs.values() if v == "down")

    if ups >= 2:
        return "🔵 Macro Tightening Pulse (2/3 rising)", inputs
    if downs >= 2:
        return "🟢 Macro Relief Pulse (2/3 falling)", inputs
    return "⚪ Macro Mixed / Neutral", inputs

def volatility_expansion(hist: pd.DataFrame, window: int = 6, lookback: int = 12):
    """
    Detect whether emotion volatility (std dev of score) is expanding over the last `lookback` points
    compared to earlier points. Returns (flag, text).
    """
    if hist is None or hist.empty or "score" not in hist.columns:
        return False, "○ Emotion volatility: insufficient data"

    score_series = pd.to_numeric(hist["score"], errors="coerce").dropna()
    if score_series is None or len(score_series) < window + lookback + 2:
        return False, "○ Emotion volatility: insufficient data"

    vol = score_series.rolling(window).std().dropna()
    if len(vol) < lookback + 1:
        return False, "○ Emotion volatility: insufficient data"

    recent = vol.iloc[-lookback:]
    delta = float(recent.iloc[-1] - recent.iloc[0])

    if delta > 0:
        return True, "🔶 Volatility Expanding (emotion acceleration)"
    return False, "✅ Volatility Stable/Contracting"


def pct_change(latest, prev):
    if prev == 0 or pd.isna(prev):
        return None
    return (latest - prev) / prev * 100.0

def fmt_pct(p):
    if p is None or pd.isna(p):
        return "n/a"
    return f"{p:+.2f}%"


   


def now_local() -> datetime:
    return datetime.now(timezone.utc).astimezone(LOCAL_TZ)


def ensure_history_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(HISTORY_PATH):
       df = pd.DataFrame(columns=[
           "ts_iso", "score", "headline_count", "panic_hits", "vix_mentions",
           "regime", "divergence", "macro_pulse", "vol_expanding"
       ])       
       df.to_csv(HISTORY_PATH, index=False)


def load_history_last_24h() -> pd.DataFrame:
    ensure_history_file()
    df = pd.read_csv(HISTORY_PATH)
    # Backwards-compatible: ensure new Decision Engine columns exist
    for col in ["regime", "divergence", "macro_pulse", "vol_expanding"]:
        if col not in df.columns:
            df[col] = None

    
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True).dt.tz_convert(LOCAL_TZ)
    cutoff = now_local() - timedelta(hours=24)
    df = df[df["ts"] >= cutoff].copy()
    df.sort_values("ts", inplace=True)
    return df


def save_point(
    score: float,
    headline_count: int,
    panic_hits: int,
    vix_mentions: int,
    regime: str | None = None,
    divergence: str | None = None,
    macro_pulse: str | None = None,
    vol_expanding: bool | None = None,
):
    ensure_history_file()
    df = pd.read_csv(HISTORY_PATH)
    ts = now_local()

    if not df.empty:
        last_ts = pd.to_datetime(df.iloc[-1]["ts_iso"], utc=True).tz_convert(LOCAL_TZ)
        if (ts - last_ts) < timedelta(minutes=4):
            return

    new_row = {
       "ts_iso": ts.astimezone(timezone.utc).isoformat(),
       "score": float(score),
       "headline_count": int(headline_count),
       "panic_hits": int(panic_hits),
       "vix_mentions": int(vix_mentions),

       # Decision Engine fields (optional)
       "regime": regime,
       "divergence": divergence,
       "macro_pulse": macro_pulse,
       "vol_expanding": vol_expanding,
  }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True).dt.tz_convert(LOCAL_TZ)
    cutoff = now_local() - timedelta(hours=24)
    df = df[df["ts"] >= cutoff].drop(columns=["ts"])
    df.to_csv(HISTORY_PATH, index=False)

def recency_weight(published_utc, half_life_hours: float) -> float:
    # Defensive defaults
    if half_life_hours is None or half_life_hours <= 0:
        half_life_hours = 6.0

    # Coerce published_utc into an aware UTC datetime
    if published_utc is None:
        return 0.0

    # If it's a string, try parse
    if isinstance(published_utc, str):
        try:
            published_utc = pd.to_datetime(published_utc, utc=True).to_pydatetime()
        except Exception:
            return 0.0

    # If datetime is naive, assume it is UTC (common RSS/API case)
    if getattr(published_utc, "tzinfo", None) is None:
        published_utc = published_utc.replace(tzinfo=timezone.utc)
    else:
        published_utc = published_utc.astimezone(timezone.utc)

    now_utc = datetime.now(timezone.utc)
    age_hours = (now_utc - published_utc).total_seconds() / 3600.0

    # Guard against future timestamps or extreme ages
    if age_hours < 0:
        age_hours = 0.0
    if age_hours > 24 * 14:  # older than 14 days -> effectively irrelevant
        return 0.0

    w = 0.5 ** (age_hours / half_life_hours)

    # Avoid printing as 0.00 for fresh items due to float tiny-ness
    if w < 1e-6:
        return 0.0
    return float(w)


def is_finance_relevant(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in FINANCE_KEYWORDS)


def count_keyword_hits(headlines, keyword_list):
    hits = 0
    for h in headlines:
        t = h.lower()
        if any(k in t for k in keyword_list):
            hits += 1
    return hits


def vix_mentions_count(headlines):
    return sum(1 for h in headlines if "vix" in h.lower() or "volatility index" in h.lower())


def fetch_headlines_finnhub(api_key: str, max_items_per_category: int = 50):
    categories = ["general", "forex", "crypto"]
    items = []

    for cat in categories:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": cat, "token": api_key}
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        for it in data[:max_items_per_category]:
            headline = it.get("headline", "").strip()
            if not headline:
                continue

            ts_unix = it.get("datetime", None)
            if ts_unix is None:
                continue

            published_utc = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc)
            items.append({
                "headline": headline,
                "source": it.get("source", ""),
                "url": it.get("url", ""),
                "published_utc": published_utc,
                "category": cat,
            })

    seen = set()
    unique = []
    for it in items:
        key = it["headline"].lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)

    # unique = [it for it in unique if is_finance_relevant(it["headline"])]
    unique.sort(key=lambda x: x["published_utc"], reverse=True)
    return unique[:120]


def score_headlines(headline_items, half_life_hours: float):
    analyzer = SentimentIntensityAnalyzer()
    scored = []
    
    df = pd.DataFrame(headline_items or [])  # ✅ ADD THIS (df always exists)
    
    # Hyper-reactive mode: only score very recent headlines
    now_utc = datetime.now(timezone.utc)

    # Use the feed's latest timestamp as the anchor (prevents "no headlines" if feed is delayed)
    times = [it.get("published_utc") for it in headline_items if it.get("published_utc") is not None]
    if not times:
       return 0.0, df

    latest_ts = max(times)
    cutoff = latest_ts - timedelta(hours=24)  # DAILY window based on feed freshness

    headline_items = [it for it in headline_items if it.get("published_utc") and it["published_utc"] >= cutoff]

    # Optional: warn if feed is stale (but still compute)
    if (now_utc - latest_ts) > timedelta(hours=24):
        st.warning(
            "⚠️ News feed appears stale (latest headline is >24h old). "
            "Using most recent 24h window from feed."
        )

    if not headline_items:
        st.warning("⚠️ No headlines in the last 12 hours — feed may be stale.")
        return 0.0, df
    
    # Debug collector (reset each run)
    st.session_state["debug_times"] = []

    for it in headline_items:
        h = it["headline"]
        compound = analyzer.polarity_scores(h)["compound"]  # [-1, 1]
        w = recency_weight(it["published_utc"], half_life_hours)

        if len(st.session_state.debug_times) < 5:
            st.write("DEBUG weight:", w)

        # Collect first 5 timestamps
        if len(st.session_state["debug_times"]) < 5:
            st.session_state["debug_times"].append({
                "headline": it.get("headline"),
                "published_utc": str(it.get("published_utc")),
            })

        scored.append({**it, "compound": compound, "weight": w, "weighted_compound": compound * w})

    if not scored:
        return 0.0, pd.DataFrame()

    # Show debug once per run
    with st.expander("Debug: published_utc values", expanded=False):
        st.write(st.session_state.get("debug_times", []))

    df_scored  = pd.DataFrame(scored)
    wsum = df_scored["weight"].sum()
    avg = (df_scored["weighted_compound"].sum() / wsum) if wsum > 0 else 0.0
    score = max(-100.0, min(100.0, avg * 100.0))
    return score, df_scored

st.title("📈 Market Emotion Index (Live)")

if "driver_data" not in st.session_state:
    st.session_state.driver_data = {}

with st.sidebar:
    st.subheader("Driver Data")
    if st.button("🔄 Update Driver Data"):
        with st.spinner("Updating driver data..."):
            try:
                ndq, ndq_prev, ndq_dt = fetch_stooq_daily_latest("qqq.us")
                btc, btc_prev, btc_dt = fetch_stooq_daily_latest("btcusd")

                st.session_state.driver_data.update({
                    "ndq": ndq, "ndq_prev": ndq_prev, "ndq_dt": ndq_dt,
                    "btc": btc, "btc_prev": btc_prev, "btc_dt": btc_dt,
                    "fetched_at": datetime.now(),
                })
                st.success("Driver data updated ✅")
            except Exception as e:
                st.error(f"Driver update failed: {e}")
# ----------------------------
def dir_from_latest_prev(latest, prev, flat_band=DIR_FLAT_BAND):
    if prev is None or pd.isna(prev) or prev == 0 or latest is None or pd.isna(latest):
        return "flat"
    r = (float(latest) / float(prev)) - 1.0
    if r > flat_band:
        return "up"
    if r < -flat_band:
        return "down"
    return "flat"

def divergence_from_levels(ndq, ndq_prev, btc, btc_prev, vix, vix_prev):
    nq_dir = dir_from_latest_prev(ndq, ndq_prev)
    btc_dir = dir_from_latest_prev(btc, btc_prev)
    vix_dir = dir_from_latest_prev(vix, vix_prev)

    is_div = (nq_dir == "up") and (btc_dir == "down") and (vix_dir == "up")
    if is_div:
        return True, "⚠️ Internal Risk Divergence (NQ↑ BTC↓ VIX↑)"
    return False, "✅ No internal divergence (NQ/BTC/VIX)"
# Drivers Panel
# ----------------------------
try:
    vix, vix_prev, vix_dt = fetch_fred_latest("VIXCLS")
    y10, y10_prev, y10_dt = fetch_fred_latest("DGS10")

    dxy, dxy_prev, dxy_dt = fetch_fred_latest("DTWEXBGS")          # US Dollar Index ETF proxy
    driver_data = st.session_state.driver_data
    ndq = driver_data.get("ndq")
    ndq_prev = driver_data.get("ndq_prev")
    btc = driver_data.get("btc")
    btc_prev = driver_data.get("btc_prev")
    # --- Decision Engine: directions, divergence, macro pulse ---
    div_flag, div_text = divergence_from_levels(ndq, ndq_prev, btc, btc_prev, vix, vix_prev)

    dxy_dir = dir_from_latest_prev(dxy, dxy_prev)
    y10_dir  = dir_from_latest_prev(y10, y10_prev)
    vix_dir  = dir_from_latest_prev(vix, vix_prev)

    ups = sum(1 for x in [dxy_dir, y10_dir, vix_dir] if x == "up")
    downs = sum(1 for x in [dxy_dir, y10_dir, vix_dir] if x == "down")

    if ups >= 2:
        pulse_text = "🔵 Macro Tightening Pulse (2/3 rising)"
    elif downs >= 2:
        pulse_text = "🟢 Macro Relief Pulse (2/3 falling)"
    else:
        pulse_text = "⚪ Macro Mixed / Neutral"

    pulse_inputs = {"DXY": dxy_dir, "10Y": y10_dir, "VIX": vix_dir}
except Exception as e:
    st.error(f"Driver fetch failed: {e}")
    div_flag, div_text = False, "⚪ Divergence unavailable"
    pulse_text = "⚪ Macro Pulse unavailable"
    pulse_inputs = {}

#ndq, ndq_prev, ndq_dt = fetch_stooq_daily_latest("qqq.us")     # Nasdaq-100 ETF proxy
    #btc, btc_prev, btc_dt = fetch_stooq_daily_latest("btc-usd")     # Bitcoin USD

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("VIX", f"{vix:.2f}", f"{(vix - vix_prev):+.2f}")

    with c2:
        bps = (y10 - y10_prev) * 100
        st.metric("US 10Y", f"{y10:.2f}%", f"{bps:+.0f} bps")

    with c3:
        st.metric("DXY", f"{dxy:.2f}", fmt_pct(pct_change(dxy, dxy_prev)))

    with c4:
        if ndq is None:
            st.metric("Nasdaq", "—", "Click Update")
        else:
            st.metric("Nasdaq", f"{ndq:,.0f}", fmt_pct(pct_change(ndq, ndq_prev)))

    with c5:
        if btc is None:
            st.metric("BTC", "—", "Click Update")
        else:
            st.metric("BTC", f"${btc:,.0f}", fmt_pct(pct_change(btc, btc_prev)))
            
st.markdown("### 🧠 Decision Engine")

de1, de2 = st.columns(2)

with de1:
    st.metric("Divergence", div_text)

with de2:
    st.metric("Macro Pulse", pulse_text)



#st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")

with st.sidebar:
    st.header("Settings")
    half_life = st.slider("Recency weighting half-life (hours)", 1, 24, DEFAULT_HALF_LIFE_HOURS)
    max_items = st.slider("Max headlines per category", 10, 100, 50)

    st.divider()
    st.subheader("API Key")
    api_key = st.secrets.get("FINNHUB_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Paste Finnhub API key", type="password")

if not api_key:
    st.warning("Add your Finnhub API key in the sidebar (or Streamlit Secrets) to start.")
    st.stop()

try:
    headlines = fetch_headlines_finnhub(api_key=api_key, max_items_per_category=max_items)
    score, df_scored = score_headlines(headlines, half_life_hours=float(half_life))
except Exception as e:
    st.error(f"Couldn’t fetch or process headlines. Error: {e}")
    st.stop()
# --- Decision Engine: regime + emotion speed (vol expansion) ---
regime = label_regime(float(score))

hist = load_history_last_24h()
if "score" in hist.columns and not hist.empty:
    vol_flag, vol_text = volatility_expansion(hist)
else:
    vol_flag, vol_text = False, "⚪ Emotion volatility: insufficient data"
all_headline_texts = [h["headline"] for h in headlines]
panic_hits = count_keyword_hits(all_headline_texts, PANIC_KEYWORDS)
vix_mentions = vix_mentions_count(all_headline_texts)

save_point(
    score=score,
    headline_count=len(headlines),
    panic_hits=panic_hits,
    vix_mentions=vix_mentions,
    regime=regime,
    divergence=div_text,
    macro_pulse=pulse_text,
    vol_expanding=vol_flag,
)
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Current Emotion Score")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": " / 100"},
        gauge={
            "axis": {"range": [-100, 100]},
            "threshold": {"line": {"width": 4}, "value": score},
        },
        title={"text": "Panic ⟵   Market Emotion   ⟶ Euphoria"}
    ))
    gauge.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=0))
    st.plotly_chart(gauge, use_container_width=True)
st.markdown("### 🧠 Regime & Emotion Speed")

r1, r2 = st.columns(2)

with r1:
    st.metric("Regime", regime)

with r2:
    st.metric("Emotion Speed", vol_text)
    #
    if score <= -70:
        st.error("🚨 Extreme PANIC zone (≤ -70)")
    elif score >= 70:
        st.success("🚀 Extreme EUPHORIA zone (≥ +70)")
    elif score <= -30:
        st.warning("⚠️ Fear leaning")
    elif score >= 30:
        st.info("🙂 Optimism leaning")
    else:
        st.write("😐 Neutral / mixed mood")

    st.caption(f"Headlines used: {len(headlines)} • Panic hits: {panic_hits} • VIX mentions: {vix_mentions}")

with colB:
    st.subheader("Emotion Trend (Last 24 Hours)")
    if hist.empty:
        st.info("No history yet — leave the app running and it will build a 24h trend automatically.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["ts"], y=hist["score"], mode="lines+markers", name="Emotion Score"))
        fig.update_layout(height=350, xaxis_title="Time", yaxis_title="Score", yaxis=dict(range=[-100, 100]))
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Top Headlines Driving the Score")

if df_scored.empty:
    st.warning("No headlines available.")
    st.stop()
    
df_scored["published_local"] = df_scored["published_utc"].dt.tz_convert(LOCAL_TZ)

top_pos = df_scored.sort_values(["weighted_compound"], ascending=False).head(8)
top_neg = df_scored.sort_values(["weighted_compound"], ascending=True).head(8)

c1, c2 = st.columns(2)

def render_headlines(df, label):
    for _, row in df.iterrows():
        t = row["published_local"].strftime("%Y-%m-%d %H:%M")
        st.markdown(f"- **{row['source']}** ({t}) — {label} **{row['compound']:+.2f}** • weight {row['weight']:.2f}")
        if row["url"]:
            st.markdown(f"  - {row['url']}")

with c1:
    st.markdown("### ✅ Most Positive")
    render_headlines(top_pos, "sentiment")

with c2:
    st.markdown("### ❌ Most Negative")
    render_headlines(top_neg, "sentiment")
