import os
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dateutil import tz
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Market Emotion Index", layout="wide")

LOCAL_TZ = tz.gettz("Africa/Johannesburg")
REFRESH_SECONDS = 300  # 5 minutes
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
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna().sort_values("Date")
    latest = float(df.iloc[-1]["Close"])
    prev = float(df.iloc[-2]["Close"]) if len(df) >= 2 else float("nan")
    date_latest = df.iloc[-1]["Date"]
    return latest, prev, date_latest

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
        df = pd.DataFrame(columns=["ts_iso", "score", "headline_count", "panic_hits", "vix_mentions"])
        df.to_csv(HISTORY_PATH, index=False)


def load_history_last_24h() -> pd.DataFrame:
    ensure_history_file()
    df = pd.read_csv(HISTORY_PATH)
    if df.empty:
        return df

    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True).dt.tz_convert(LOCAL_TZ)
    cutoff = now_local() - timedelta(hours=24)
    df = df[df["ts"] >= cutoff].copy()
    df.sort_values("ts", inplace=True)
    return df


def save_point(score: float, headline_count: int, panic_hits: int, vix_mentions: int):
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
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df["ts"] = pd.to_datetime(df["ts_iso"], utc=True).dt.tz_convert(LOCAL_TZ)
    cutoff = now_local() - timedelta(hours=24)
    df = df[df["ts"] >= cutoff].drop(columns=["ts"])
    df.to_csv(HISTORY_PATH, index=False)


def recency_weight(published_utc: datetime, half_life_hours: float) -> float:
    age = (datetime.now(timezone.utc) - published_utc).total_seconds() / 3600.0
    return 0.5 ** (age / half_life_hours)


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


@st.cache_data(ttl=REFRESH_SECONDS)
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

    unique = [it for it in unique if is_finance_relevant(it["headline"])]
    unique.sort(key=lambda x: x["published_utc"], reverse=True)
    return unique[:120]


def score_headlines(headline_items, half_life_hours: float):
    analyzer = SentimentIntensityAnalyzer()
    scored = []

    for it in headline_items:
        h = it["headline"]
        compound = analyzer.polarity_scores(h)["compound"]  # [-1, 1]
        w = recency_weight(it["published_utc"], half_life_hours)
        scored.append({**it, "compound": compound, "weight": w, "weighted_compound": compound * w})

    if not scored:
        return 0.0, pd.DataFrame()

    df = pd.DataFrame(scored)
    wsum = df["weight"].sum()
    avg = (df["weighted_compound"].sum() / wsum) if wsum > 0 else 0.0
    score = max(-100.0, min(100.0, avg * 100.0))
    return score, df


st.title("📈 Market Emotion Index (Live)")
# ----------------------------
# Drivers Panel
# ----------------------------
try:
     vix, vix_prev, vix_dt = fetch_fred_latest("VIXCLS")
    y10, y10_prev, y10_dt = fetch_fred_latest("DGS10")

    dxy, dxy_prev, dxy_dt = fetch_stooq_daily_latest("dx-y.nyse")   # US Dollar Index ETF proxy
    ndq, ndq_prev, ndq_dt = fetch_stooq_daily_latest("qqq.us")      # Nasdaq-100 ETF proxy
    btc, btc_prev, btc_dt = fetch_stooq_daily_latest("btcusd")      # Bitcoin USD

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("VIX", f"{vix:.2f}", f"{(vix - vix_prev):+.2f}")

    with c2:
        bps = (y10 - y10_prev) * 100
        st.metric("US 10Y", f"{y10:.2f}%", f"{bps:+.0f} bps")

    with c3:
        st.metric("DXY", f"{dxy:.2f}", fmt_pct(pct_change(dxy, dxy_prev)))

    with c4:
        st.metric("Nasdaq", f"{ndq:,.0f}", fmt_pct(pct_change(ndq, ndq_prev)))

    with c5:
        st.metric("BTC", f"${btc:,.0f}", fmt_pct(pct_change(btc, btc_prev)))

    st.divider()

except Exception as e:
    st.warning(f"Driver data loading... ({e})")


st_autorefresh(interval=REFRESH_SECONDS * 1000, key="refresh")

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

all_headline_texts = [h["headline"] for h in headlines]
panic_hits = count_keyword_hits(all_headline_texts, PANIC_KEYWORDS)
vix_mentions = vix_mentions_count(all_headline_texts)

save_point(score=score, headline_count=len(headlines), panic_hits=panic_hits, vix_mentions=vix_mentions)
hist = load_history_last_24h()

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
    st.warning("No finance-relevant headlines matched the filter. Try increasing max headlines.")
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
