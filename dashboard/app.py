"""Streamlit dashboard — Fake Review Detection capstone.

Pages:
  1. Model Results       — metrics, D3 confusion-matrix heatmaps, D3 feature importance
  2. EDA Summary         — 6 report/figures/ PNGs with captions
  3. Live Stream Monitor — auto-refreshing parquet sink view
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "report" / "figures"
SINK_DIR    = ROOT / "data" / "stream_sink" / "predictions"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — only overrides config.toml cannot reach ─────────────────────────────
st.markdown("""
<style>
/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* Content max-width & spacing */
.block-container {
    max-width: 1160px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Page title */
h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #1d1d1f !important;
    margin-bottom: 0 !important;
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: #1d1d1f !important;
    margin-top: 2.4rem !important;
    margin-bottom: 0.4rem !important;
}
h3 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #1d1d1f !important;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e5e5e7;
    margin: 2rem 0;
}

/* Caption / secondary text */
[data-testid="stCaptionContainer"] p,
small {
    color: #6e6e73 !important;
    font-size: 0.82rem !important;
    line-height: 1.6;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f5f5f7;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e5e5e7;
}
[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6e6e73 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #1d1d1f !important;
    letter-spacing: -0.02em;
}

/* Dataframe */
[data-testid="stDataFrame"] iframe {
    border-radius: 10px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid #e5e5e7;
    background: #f5f5f7;
}
section[data-testid="stSidebar"] > div { padding-top: 2rem; }

/* Radio nav items */
div[role="radiogroup"] > label {
    padding: 0.45rem 0.6rem !important;
    border-radius: 8px;
    margin-bottom: 2px;
    font-size: 0.9rem !important;
    color: #1d1d1f !important;
}
div[role="radiogroup"] > label:hover { background: #e8e8ed; }

/* Alert / info boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: 1px solid #e5e5e7 !important;
    border-left: 3px solid #0071e3 !important;
    background: #f0f6ff !important;
    color: #1d1d1f !important;
}
[data-testid="stAlert"] p { color: #1d1d1f !important; font-size: 0.88rem !important; }

/* Code blocks */
code {
    background: #f5f5f7 !important;
    color: #1d1d1f !important;
    border-radius: 5px;
    font-size: 0.82rem;
}
pre {
    background: #f5f5f7 !important;
    border: 1px solid #e5e5e7 !important;
    border-radius: 10px !important;
}

/* Toggle */
[data-testid="stToggle"] label { color: #1d1d1f !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# D3 chart components
# ═════════════════════════════════════════════════════════════════════════════

def d3_bar_chart(series: pd.Series, accent: str = "#0071e3") -> None:
    """Horizontal feature-importance bar chart rendered with D3 v7."""
    data = (
        series
        .sort_values(ascending=True)
        .reset_index()
        .set_axis(["feature", "importance"], axis=1)
        .to_dict(orient="records")
    )
    n          = len(data)
    bar_h      = 30
    margin     = {"top": 8, "right": 72, "bottom": 28, "left": 200}
    inner_h    = n * bar_h
    total_h    = inner_h + margin["top"] + margin["bottom"]
    data_json  = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: -apple-system, BlinkMacSystemFont,
          "SF Pro Text", "Segoe UI", sans-serif; overflow: hidden; }}
  .tick line {{ stroke: #e5e5e7; }}
  .domain    {{ stroke: #e5e5e7; }}
  .bar       {{ transition: opacity .15s; }}
  .bar:hover {{ opacity: .75; cursor: default; }}
  #tooltip {{
    position: absolute; pointer-events: none;
    background: #1d1d1f; color: #fff;
    font-size: 11px; border-radius: 6px;
    padding: 5px 9px; opacity: 0;
    transition: opacity .12s;
    white-space: nowrap;
  }}
</style>
</head>
<body>
<div id="tooltip"></div>
<svg id="chart" width="100%" height="{total_h}"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const DATA   = {data_json};
const ACCENT = "{accent}";
const margin = {{ top: {margin["top"]}, right: {margin["right"]},
                  bottom: {margin["bottom"]}, left: {margin["left"]} }};

const svg  = d3.select("#chart");
const W    = document.getElementById("chart").clientWidth;
const iW   = W - margin.left - margin.right;
const iH   = {inner_h};

const g = svg.append("g")
    .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

const x = d3.scaleLinear()
    .domain([0, d3.max(DATA, d => d.importance) * 1.1])
    .range([0, iW]).nice();

const y = d3.scaleBand()
    .domain(DATA.map(d => d.feature))
    .range([0, iH])
    .padding(0.35);

// Subtle grid
g.append("g").attr("class", "grid")
    .attr("transform", `translate(0,${{iH}})`)
    .call(d3.axisBottom(x).ticks(4).tickSize(-iH).tickFormat(""))
    .call(g => g.select(".domain").remove())
    .call(g => g.selectAll("line")
        .attr("stroke", "#e5e5e7").attr("stroke-dasharray", "3,3"));

// Bars
g.selectAll(".bar")
    .data(DATA).join("rect")
    .attr("class", "bar")
    .attr("y", d => y(d.feature))
    .attr("x", 0)
    .attr("height", y.bandwidth())
    .attr("rx", 4).attr("ry", 4)
    .attr("fill", ACCENT)
    .attr("width", 0)
    .on("mouseover", (event, d) => {{
        const tip = document.getElementById("tooltip");
        tip.style.opacity = 1;
        tip.innerHTML = `<b>${{d.feature}}</b>&nbsp;&nbsp;${{d3.format(".5f")(d.importance)}}`;
    }})
    .on("mousemove", event => {{
        const tip = document.getElementById("tooltip");
        tip.style.left = (event.pageX + 12) + "px";
        tip.style.top  = (event.pageY - 22) + "px";
    }})
    .on("mouseout", () => {{
        document.getElementById("tooltip").style.opacity = 0;
    }})
    .transition().duration(550).ease(d3.easeCubicOut)
    .attr("width", d => x(d.importance));

// Value labels
g.selectAll(".val")
    .data(DATA).join("text")
    .attr("class", "val")
    .attr("x", d => x(d.importance) + 6)
    .attr("y", d => y(d.feature) + y.bandwidth() / 2)
    .attr("dy", "0.35em")
    .attr("fill", "#6e6e73")
    .attr("font-size", "11px")
    .text(d => d3.format(".4f")(d.importance));

// Y axis
g.append("g")
    .call(d3.axisLeft(y).tickSize(0).tickPadding(10))
    .call(g => g.select(".domain").remove())
    .selectAll("text")
    .attr("fill", "#1d1d1f")
    .attr("font-size", "12px");

// X axis
g.append("g")
    .attr("transform", `translate(0,${{iH}})`)
    .call(d3.axisBottom(x).ticks(4).tickFormat(d3.format(".2f")))
    .call(g => g.select(".domain").attr("stroke", "#e5e5e7"))
    .selectAll("text")
    .attr("fill", "#6e6e73").attr("font-size", "11px");
</script>
</body>
</html>"""
    components.html(html, height=total_h + 16, scrolling=False)


def d3_confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> None:
    """2×2 D3 heatmap confusion matrix."""
    total    = tn + fp + fn + tp
    cells    = [
        {"r": 0, "c": 0, "v": tn, "pct": tn / total, "lbl": "True Negative",  "ok": True},
        {"r": 0, "c": 1, "v": fp, "pct": fp / total, "lbl": "False Positive",  "ok": False},
        {"r": 1, "c": 0, "v": fn, "pct": fn / total, "lbl": "False Negative",  "ok": False},
        {"r": 1, "c": 1, "v": tp, "pct": tp / total, "lbl": "True Positive",   "ok": True},
    ]
    data_json = json.dumps(cells)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #ffffff; font-family: -apple-system, BlinkMacSystemFont,
          "SF Pro Text", "Segoe UI", sans-serif; overflow: hidden; }}
</style>
</head>
<body>
<svg id="cm"></svg>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const CELLS   = {data_json};
const cell    = 130;
const pad     = {{ top: 44, left: 72, right: 16, bottom: 44 }};
const W       = cell * 2 + pad.left + pad.right;
const H       = cell * 2 + pad.top  + pad.bottom;

const okColor  = d3.scaleLinear().domain([0, 1]).range(["#e8f5e9", "#34c759"]);
const badColor = d3.scaleLinear().domain([0, 1]).range(["#fff3f3", "#ff3b30"]);
const maxVal   = d3.max(CELLS, d => d.v);

const svg = d3.select("#cm")
    .attr("width", W).attr("height", H);

const g = svg.append("g")
    .attr("transform", `translate(${{pad.left}},${{pad.top}})`);

// Cells
CELLS.forEach(d => {{
    const x     = d.c * cell;
    const y     = d.r * cell;
    const scale = d.v / maxVal;
    const fill  = d.ok ? okColor(scale) : badColor(scale);
    const dark  = scale > 0.55;

    g.append("rect")
        .attr("x", x).attr("y", y)
        .attr("width", cell - 2).attr("height", cell - 2)
        .attr("rx", 8).attr("fill", fill)
        .attr("stroke", "#e5e5e7").attr("stroke-width", 1);

    // Count
    g.append("text")
        .attr("x", x + cell / 2 - 1).attr("y", y + cell / 2 - 8)
        .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
        .attr("fill", dark ? "#ffffff" : "#1d1d1f")
        .attr("font-size", "20px").attr("font-weight", "700")
        .attr("font-family", "-apple-system, sans-serif")
        .text(d3.format(",")(d.v));

    // Percentage
    g.append("text")
        .attr("x", x + cell / 2 - 1).attr("y", y + cell / 2 + 14)
        .attr("text-anchor", "middle").attr("dominant-baseline", "middle")
        .attr("fill", dark ? "rgba(255,255,255,.75)" : "#6e6e73")
        .attr("font-size", "12px")
        .attr("font-family", "-apple-system, sans-serif")
        .text(d3.format(".1%")(d.pct));

    // Label
    g.append("text")
        .attr("x", x + cell / 2 - 1).attr("y", y + cell - 12)
        .attr("text-anchor", "middle")
        .attr("fill", dark ? "rgba(255,255,255,.6)" : "#6e6e73")
        .attr("font-size", "10px").attr("font-weight", "500")
        .attr("text-transform", "uppercase")
        .attr("font-family", "-apple-system, sans-serif")
        .text(d.lbl.toUpperCase());
}});

// Column headers (Predicted)
const colLabels = ["Predicted: Legit", "Predicted: Fraud"];
colLabels.forEach((lbl, i) => {{
    g.append("text")
        .attr("x", i * cell + cell / 2 - 1).attr("y", -16)
        .attr("text-anchor", "middle")
        .attr("fill", "#6e6e73").attr("font-size", "11px").attr("font-weight", "500")
        .attr("font-family", "-apple-system, sans-serif")
        .text(lbl);
}});

// Row headers (Actual)
const rowLabels = ["Actual: Legit", "Actual: Fraud"];
rowLabels.forEach((lbl, i) => {{
    g.append("text")
        .attr("x", -10).attr("y", i * cell + cell / 2 - 1)
        .attr("text-anchor", "end").attr("dominant-baseline", "middle")
        .attr("fill", "#6e6e73").attr("font-size", "11px").attr("font-weight", "500")
        .attr("font-family", "-apple-system, sans-serif")
        .text(lbl);
}});
</script>
</body>
</html>"""
    components.html(html, height=2 * 130 + 44 + 44 + 16, scrolling=False)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Fake Review Detection")
    st.caption("CS-GY 6513 · Big Data · Spring 2026")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Model Results", "EDA Summary", "Live Stream Monitor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "Labels are weak-supervision proxies — not verified ground truth. "
        "Model metrics measure heuristic-pattern learning."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Model Results
# ═════════════════════════════════════════════════════════════════════════════
if page == "Model Results":
    st.markdown("# Model Results")
    st.caption(
        "Random Forest · Spark MLlib · 8.2M labeled Amazon reviews · "
        "December 2014 – August 2015 · Wireless · Books · Apparel"
    )
    st.markdown("---")

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("## Evaluation Metrics")
    st.caption("Full model (19 features) vs. leakage-controlled (12 features, reviewer-behaviour removed).")

    METRICS = {
        "Accuracy":           (0.9870, 0.9131),
        "Weighted Precision":  (0.9871, 0.8976),
        "Weighted Recall":     (0.9870, 0.9131),
        "Weighted F1":         (0.9870, 0.8720),
        "ROC AUC":             (0.9917, 0.7098),
        "Area Under PR":       (0.9676, 0.2058),
    }

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("#### `rf_v1` — 19 features")
        df_v1 = pd.DataFrame(
            {"Metric": list(METRICS), "Value": [v[0] for v in METRICS.values()]}
        ).set_index("Metric")
        st.dataframe(df_v1.style.format({"Value": "{:.4f}"}), use_container_width=True)

    with col_r:
        st.markdown("#### `rf_no_leakage_v1` — 12 features")
        df_nl = pd.DataFrame(
            {"Metric": list(METRICS), "Value": [v[1] for v in METRICS.values()]}
        ).set_index("Metric")
        st.dataframe(df_nl.style.format({"Value": "{:.4f}"}), use_container_width=True)

    st.info(
        "**Why the gap?**  The full model's ROC AUC of 0.99 is expected — its top "
        "features (*max reviews per day*, *total reviews*, *rating stddev*) mirror the "
        "heuristics used to generate labels. The leakage-controlled model (ROC AUC 0.71) "
        "excludes those features for a more honest generalisation estimate. "
        "Both are reported for transparency."
    )

    st.markdown("---")

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("## Confusion Matrices")
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("#### `rf_v1`")
        d3_confusion_matrix(tn=1_554_633, fp=11_690, fn=10_591, tp=138_807)
    with col_r:
        st.markdown("#### `rf_no_leakage_v1`")
        d3_confusion_matrix(tn=1_566_144, fp=179, fn=148_902, tp=496)

    st.markdown("---")

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("## Feature Importance")
    st.caption("Relative contribution of each feature to Random Forest split decisions.")

    fi_v1 = pd.Series({
        "rev_max_per_day":                   0.488322,
        "rev_total_reviews":                 0.306323,
        "rev_rating_stddev":                 0.080964,
        "xf_rating_vs_reviewer":             0.051213,
        "rev_days_active":                   0.028554,
        "txt_word_count":                    0.011281,
        "xf_rating_vs_product":              0.007910,
        "txt_body_length":                   0.007833,
        "prod_pct_5star":                    0.005164,
        "rev_pct_verified":                  0.004802,
        "prod_avg_rating":                   0.003557,
        "txt_sentiment":                     0.001650,
        "prod_pct_verified":                 0.000907,
        "xf_extreme_review_neutral_product": 0.000429,
        "prod_total_reviews":                0.000373,
        "prod_days_active":                  0.000285,
        "txt_caps_ratio":                    0.000223,
        "txt_caps_word_count":               0.000121,
        "txt_exclamation_count":             0.000088,
    })
    fi_nl = pd.Series({
        "txt_word_count":        0.377724,
        "txt_body_length":       0.284815,
        "prod_avg_rating":       0.073267,
        "txt_sentiment":         0.053895,
        "prod_total_reviews":    0.052221,
        "prod_pct_5star":        0.043849,
        "prod_pct_verified":     0.032261,
        "xf_rating_vs_product":  0.026542,
        "txt_caps_word_count":   0.018742,
        "txt_exclamation_count": 0.015473,
        "prod_days_active":      0.014775,
        "txt_caps_ratio":        0.006437,
    })

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("#### `rf_v1`")
        d3_bar_chart(fi_v1, accent="#0071e3")
        st.caption(
            "Reviewer-behaviour features account for 88% of total importance. "
            "These are the same signals used to define the labels."
        )
    with col_r:
        st.markdown("#### `rf_no_leakage_v1`")
        d3_bar_chart(fi_nl, accent="#34c759")
        st.caption(
            "Text length and word count dominate — genuine linguistic signals "
            "independent of the labelling heuristic."
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Summary
# ═════════════════════════════════════════════════════════════════════════════
elif page == "EDA Summary":
    st.markdown("# Exploratory Data Analysis")
    st.caption(
        "8.2M reviews · December 2014 – August 2015 · "
        "Wireless · Books · Apparel"
    )
    st.markdown("---")

    FIGURES = [
        (
            "rating_distribution.png",
            "Rating Distribution by Category",
            "Ratings are heavily skewed positive — ~57% of reviews are 5-star. "
            "Books has the highest 5-star volume. Wireless shows the sharpest 1-star "
            "spike relative to mid-range ratings, consistent with literature on "
            "consumer-electronics review fraud.",
        ),
        (
            "helpful_vote_ratio.png",
            "Helpful-Vote Ratio Distribution",
            "Strongly bimodal: reviews cluster near ratio ≈ 1.0 (clearly useful) or "
            "≈ 0.0 (clearly unhelpful). Reviews with helpful / total < 0.20 and ≥ 5 "
            "votes form weak-supervision signal H1, firing on 0.38% of the dataset.",
        ),
        (
            "reviewer_activity.png",
            "Reviews per Reviewer (log scale)",
            "Classic power-law distribution. The vast majority of customers post exactly "
            "one review. The long tail of reviewers with 50+ reviews in 8 months is "
            "disproportionately active — a known fraud-recruitment pattern.",
        ),
        (
            "verified_vs_unverified.png",
            "Verified vs Unverified Purchases by Category",
            "Apparel ~96% verified · Wireless ~92% · Books ~79%. Books' high unverified "
            "rate reflects digital and gifted purchases. Unverified reviews concentrate "
            "disproportionately in extreme ratings — a documented fraud signal.",
        ),
        (
            "review_velocity_timeseries.png",
            "Daily Review Volume · December 2014 – August 2015",
            "Volume oscillates 30k–38k reviews/day with a weekly rhythm. A single-day "
            "spike to ~51k in mid-June 2015 stands ~35% above baseline with no run-up — "
            "a strong candidate for batch-fraud activity.",
        ),
        (
            "top_reviewers.png",
            "Top 20 Most Prolific Reviewers",
            "The most prolific reviewer posted ~1,270 reviews across 8 months (~5.2/day). "
            "The top-20 cutoff is ~295 reviews. Rates of 3+ reviews/day sustained over "
            "months are rare among genuine independent reviewers.",
        ),
    ]

    for filename, title, caption in FIGURES:
        path = FIGURES_DIR / filename
        st.markdown(f"## {title}")
        if path.exists():
            st.image(str(path), use_column_width=True)
        else:
            st.warning(f"Figure not found: `report/figures/{filename}`")
        st.caption(caption)
        st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Stream Monitor
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Stream Monitor":
    st.markdown("# Live Stream Monitor")
    top_l, top_r = st.columns([5, 1])
    with top_r:
        auto_refresh = st.toggle("Auto-refresh", value=True)

    def load_predictions() -> pd.DataFrame:
        if not SINK_DIR.exists():
            return pd.DataFrame()
        parts = list(SINK_DIR.rglob("*.parquet"))
        if not parts:
            return pd.DataFrame()
        return pd.read_parquet(SINK_DIR, engine="pyarrow")

    df = load_predictions()

    if df.empty:
        with top_l:
            st.caption("No data yet. Start the producer and consumer to populate this page.")
        st.markdown("---")
        st.markdown("#### How to start streaming")
        st.code("""\
# Terminal 1 — Kafka broker
docker compose -f docker/docker-compose.yml up -d

# Terminal 2 — consumer  (writes predictions → data/stream_sink/)
python streaming/consumer.py

# Terminal 3 — producer  (sends reviews → Kafka)
python streaming/producer.py""", language="bash")

    else:
        total      = len(df)
        n_fraud    = int((df["prediction"] == 1.0).sum())
        fraud_rate = n_fraud / total if total else 0.0

        with top_l:
            st.caption(f"{total:,} predictions in sink")

        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Scored",     f"{total:,}")
        m2.metric("Flagged as Fraud", f"{n_fraud:,}")
        m3.metric("Fraud Rate",       f"{fraud_rate:.1%}")

        st.markdown("---")
        st.markdown("## Recent Predictions")

        display_cols = [c for c in
            ["timestamp", "review_id", "product_id", "product_category",
             "star_rating", "fraud_probability", "prediction"]
            if c in df.columns]

        recent = df[display_cols].copy()
        if "timestamp" in recent.columns:
            recent = recent.sort_values("timestamp", ascending=False)
        if "fraud_probability" in recent.columns:
            recent["fraud_probability"] = recent["fraud_probability"].round(4)
        if "prediction" in recent.columns:
            recent["prediction"] = recent["prediction"].map({0.0: "Legit", 1.0: "Fraud"})

        st.dataframe(recent.head(50), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("## Fraud Alerts")
        st.caption("Products with 3 or more fraud flags in the current sink window.")

        fraud_rows = df[df["prediction"] == 1.0].copy() if "prediction" in df.columns else pd.DataFrame()

        if not fraud_rows.empty and "product_id" in fraud_rows.columns:
            group_cols = [c for c in ["product_id", "product_category"] if c in fraud_rows.columns]
            agg = (
                fraud_rows.groupby(group_cols)
                .agg(fraud_count=("prediction", "count"),
                     avg_fraud_prob=("fraud_probability", "mean"))
                .reset_index()
                .query("fraud_count >= 3")
                .sort_values("fraud_count", ascending=False)
            )
            if agg.empty:
                st.caption("No product has reached 3 fraud flags yet.")
            else:
                agg["avg_fraud_prob"] = agg["avg_fraud_prob"].round(4)
                st.dataframe(agg, use_container_width=True, hide_index=True)
        else:
            st.caption("No fraud predictions yet.")

        if "product_category" in df.columns and "prediction" in df.columns:
            st.markdown("---")
            st.markdown("## Fraud Rate by Category")
            by_cat = (
                df.groupby("product_category")["prediction"]
                .agg(
                    total="count",
                    fraud=lambda x: (x == 1.0).sum(),
                    fraud_rate=lambda x: round((x == 1.0).mean(), 4),
                )
                .reset_index()
            )
            st.dataframe(by_cat, use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(5)
        st.rerun()
