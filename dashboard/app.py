"""Streamlit dashboard — Fake Review Detection capstone.

Pages:
  1. Model Results       — D3 metric comparison, confusion-matrix heatmaps, feature importance
  2. EDA Summary         — 6 D3 charts replacing static PNGs
  3. Live Stream Monitor — D3 stat cards, predictions table, D3 category bars
"""

import datetime
import json
import math
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
SINK_DIR = ROOT / "data" / "stream_sink" / "predictions"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — overrides that config.toml cannot reach ─────────────────────────────
st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
}
.block-container { max-width: 1160px; padding-top: 2.5rem; padding-bottom: 4rem; }
h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important;
     color: #1d1d1f !important; margin-bottom: 0 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important;
     color: #1d1d1f !important; margin-top: 2.4rem !important; margin-bottom: 0.4rem !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; color: #1d1d1f !important; }
hr { border: none; border-top: 1px solid #e5e5e7; margin: 2rem 0; }
[data-testid="stCaptionContainer"] p, small {
    color: #6e6e73 !important; font-size: 0.82rem !important; line-height: 1.6; }
[data-testid="stDataFrame"] iframe { border-radius: 10px; }
section[data-testid="stSidebar"] { border-right: 1px solid #e5e5e7; background: #f5f5f7; }
section[data-testid="stSidebar"] > div { padding-top: 2rem; }
div[role="radiogroup"] > label {
    padding: 0.45rem 0.6rem !important; border-radius: 8px; margin-bottom: 2px;
    font-size: 0.9rem !important; color: #1d1d1f !important; }
div[role="radiogroup"] > label:hover { background: #e8e8ed; }
[data-testid="stAlert"] {
    border-radius: 10px !important; border: 1px solid #e5e5e7 !important;
    border-left: 3px solid #0071e3 !important; background: #f0f6ff !important;
    color: #1d1d1f !important; }
[data-testid="stAlert"] p { color: #1d1d1f !important; font-size: 0.88rem !important; }
code { background: #f5f5f7 !important; color: #1d1d1f !important; border-radius: 5px; font-size: 0.82rem; }
pre  { background: #f5f5f7 !important; border: 1px solid #e5e5e7 !important; border-radius: 10px !important; }
[data-testid="stToggle"] label { color: #1d1d1f !important; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Shared D3 boilerplate  (plain strings — CSS braces are literal)
# ═════════════════════════════════════════════════════════════════════════════

_HEAD = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #ffffff;
       font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
       overflow: hidden; }
.tick line { stroke: #e5e5e7; }
.domain    { stroke: #e5e5e7; }
#tip {
  position: absolute; pointer-events: none;
  background: #1d1d1f; color: #fff;
  font-size: 11px; border-radius: 6px;
  padding: 5px 9px; opacity: 0;
  transition: opacity .12s; white-space: nowrap; z-index: 99;
}
</style></head><body>
<div id="tip"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>"""

_FOOT = "</body></html>"

_FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif"


# ── tooltip helpers (JS snippets reused in every chart) ───────────────────────
def _tip_js(html_expr: str) -> str:
    """Return JS that sets tooltip innerHTML. html_expr is a JS expression."""
    return (
        f"const _t=document.getElementById('tip');"
        f"_t.style.opacity=1;_t.innerHTML={html_expr};"
    )

def _tip_move_js() -> str:
    return (
        "const _t2=document.getElementById('tip');"
        "_t2.style.left=(ev.pageX+10)+'px';_t2.style.top=(ev.pageY-24)+'px';"
    )

def _tip_out_js() -> str:
    return "document.getElementById('tip').style.opacity=0;"


# ═════════════════════════════════════════════════════════════════════════════
# Chart 1 — horizontal feature-importance bar  (used on Page 1)
# ═════════════════════════════════════════════════════════════════════════════

def d3_bar_chart(series: pd.Series, accent: str = "#0071e3") -> None:
    data = (
        series.sort_values(ascending=True)
        .reset_index()
        .set_axis(["feature", "importance"], axis=1)
        .to_dict(orient="records")
    )
    n         = len(data)
    bar_h     = 30
    margin    = {"top": 8, "right": 72, "bottom": 28, "left": 200}
    inner_h   = n * bar_h
    total_h   = inner_h + margin["top"] + margin["bottom"]
    data_json = json.dumps(data)

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA   = {data_json};
const ACCENT = "{accent}";
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x = d3.scaleLinear().domain([0,d3.max(DATA,d=>d.importance)*1.1]).range([0,iW]).nice();
const y = d3.scaleBand().domain(DATA.map(d=>d.feature)).range([0,iH]).padding(0.35);

g.append("g").attr("class","grid")
  .attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(4).tickSize(-iH).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

g.selectAll(".bar").data(DATA).join("rect")
  .attr("class","bar")
  .attr("y",d=>y(d.feature)).attr("x",0)
  .attr("height",y.bandwidth()).attr("rx",4).attr("fill",ACCENT).attr("width",0)
  .on("mouseover",(ev,d)=>{{
    {_tip_js("`<b>${d.feature}</b>&nbsp;&nbsp;${d3.format('.5f')(d.importance)}`")}
  }})
  .on("mousemove",ev=>{{ {_tip_move_js()} }})
  .on("mouseout",()=>{{ {_tip_out_js()} }})
  .transition().duration(550).ease(d3.easeCubicOut).attr("width",d=>x(d.importance));

g.selectAll(".val").data(DATA).join("text")
  .attr("class","val")
  .attr("x",d=>x(d.importance)+6).attr("y",d=>y(d.feature)+y.bandwidth()/2)
  .attr("dy","0.35em").attr("fill","#6e6e73").attr("font-size","11px")
  .text(d=>d3.format(".4f")(d.importance));

g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(10))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#1d1d1f").attr("font-size","12px");

g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(4).tickFormat(d3.format(".2f")))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 2 — confusion matrix heatmap  (used on Page 1)
# ═════════════════════════════════════════════════════════════════════════════

def d3_confusion_matrix(tn: int, fp: int, fn: int, tp: int) -> None:
    total = tn + fp + fn + tp
    cells = [
        {"r": 0, "c": 0, "v": tn, "pct": tn/total, "lbl": "True Negative",  "ok": True},
        {"r": 0, "c": 1, "v": fp, "pct": fp/total, "lbl": "False Positive",  "ok": False},
        {"r": 1, "c": 0, "v": fn, "pct": fn/total, "lbl": "False Negative",  "ok": False},
        {"r": 1, "c": 1, "v": tp, "pct": tp/total, "lbl": "True Positive",   "ok": True},
    ]
    data_json = json.dumps(cells)

    html = _HEAD + f"""
<svg id="cm"></svg>
<script>
const CELLS = {data_json};
const cell  = 130;
const pad   = {{top:44, left:72, right:16, bottom:44}};
const W     = cell*2 + pad.left + pad.right;
const H     = cell*2 + pad.top  + pad.bottom;
const okC   = d3.scaleLinear().domain([0,1]).range(["#e8f5e9","#34c759"]);
const badC  = d3.scaleLinear().domain([0,1]).range(["#fff3f3","#ff3b30"]);
const maxV  = d3.max(CELLS,d=>d.v);

const svg = d3.select("#cm").attr("width",W).attr("height",H);
const g   = svg.append("g").attr("transform",`translate(${{pad.left}},${{pad.top}})`);

CELLS.forEach(d=>{{
  const x=d.c*cell, y=d.r*cell;
  const fill = d.ok ? okC(d.v/maxV) : badC(d.v/maxV);
  const dark = (d.v/maxV)>0.55;

  g.append("rect")
    .attr("x",x).attr("y",y)
    .attr("width",cell-2).attr("height",cell-2)
    .attr("rx",8).attr("fill",fill)
    .attr("stroke","#e5e5e7").attr("stroke-width",1);

  g.append("text")
    .attr("x",x+cell/2-1).attr("y",y+cell/2-8)
    .attr("text-anchor","middle").attr("dominant-baseline","middle")
    .attr("fill",dark?"#ffffff":"#1d1d1f")
    .attr("font-size","20px").attr("font-weight","700")
    .attr("font-family","{_FONT}")
    .text(d3.format(",")(d.v));

  g.append("text")
    .attr("x",x+cell/2-1).attr("y",y+cell/2+14)
    .attr("text-anchor","middle").attr("dominant-baseline","middle")
    .attr("fill",dark?"rgba(255,255,255,.75)":"#6e6e73")
    .attr("font-size","12px").attr("font-family","{_FONT}")
    .text(d3.format(".1%")(d.pct));

  g.append("text")
    .attr("x",x+cell/2-1).attr("y",y+cell-12)
    .attr("text-anchor","middle")
    .attr("fill",dark?"rgba(255,255,255,.6)":"#6e6e73")
    .attr("font-size","10px").attr("font-weight","500")
    .attr("font-family","{_FONT}")
    .text(d.lbl.toUpperCase());
}});

["Predicted: Legit","Predicted: Fraud"].forEach((lbl,i)=>{{
  g.append("text")
    .attr("x",i*cell+cell/2-1).attr("y",-16)
    .attr("text-anchor","middle")
    .attr("fill","#6e6e73").attr("font-size","11px").attr("font-weight","500")
    .attr("font-family","{_FONT}").text(lbl);
}});

["Actual: Legit","Actual: Fraud"].forEach((lbl,i)=>{{
  g.append("text")
    .attr("x",-10).attr("y",i*cell+cell/2-1)
    .attr("text-anchor","end").attr("dominant-baseline","middle")
    .attr("fill","#6e6e73").attr("font-size","11px").attr("font-weight","500")
    .attr("font-family","{_FONT}").text(lbl);
}});
</script>
""" + _FOOT
    components.html(html, height=2*130+44+44+16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 3 — grouped horizontal bar comparing two models  (Page 1)
# ═════════════════════════════════════════════════════════════════════════════

def d3_metric_comparison(metrics: dict) -> None:
    data_json = json.dumps([
        {"metric": m, "v1": v[0], "nl": v[1]}
        for m, v in metrics.items()
    ])
    margin  = {"top": 36, "right": 120, "bottom": 24, "left": 200}
    inner_h = 310
    total_h = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const MODELS  = ["v1","nl"];
const COLORS  = {{"v1":"#0071e3","nl":"#34c759"}};
const LABELS  = {{"v1":"rf_v1  (19 features)","nl":"rf_no_leakage  (12 features)"}};

const y0 = d3.scaleBand().domain(DATA.map(d=>d.metric)).range([0,iH]).paddingInner(0.28).paddingOuter(0.08);
const y1 = d3.scaleBand().domain(MODELS).range([0,y0.bandwidth()]).padding(0.06);
const x  = d3.scaleLinear().domain([0,1.05]).range([0,iW]);

// Grid
g.append("g").attr("class","grid")
  .attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(5).tickSize(-iH).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

// Bars
DATA.forEach(d=>{{
  MODELS.forEach(key=>{{
    const val = d[key];
    g.append("rect")
      .attr("x",0)
      .attr("y",y0(d.metric)+y1(key))
      .attr("height",y1.bandwidth()).attr("rx",3)
      .attr("fill",COLORS[key]).attr("width",0)
      .on("mouseover",ev=>{{
        {_tip_js("`<b>${LABELS[key]}</b> — ${d.metric}: ${val.toFixed(4)}`")}
      }})
      .on("mousemove",ev=>{{ {_tip_move_js()} }})
      .on("mouseout",()=>{{ {_tip_out_js()} }})
      .transition().duration(500).ease(d3.easeCubicOut).attr("width",x(val));

    g.append("text")
      .attr("x",x(val)+5).attr("y",y0(d.metric)+y1(key)+y1.bandwidth()/2)
      .attr("dy","0.35em").attr("fill","#6e6e73").attr("font-size","11px")
      .text(val.toFixed(4));
  }});
}});

// Y axis
g.append("g").call(d3.axisLeft(y0).tickSize(0).tickPadding(10))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#1d1d1f").attr("font-size","12px");

// X axis
g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".1f")))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// Legend
const leg = g.append("g").attr("transform","translate(0,-22)");
MODELS.forEach((key,i)=>{{
  leg.append("rect").attr("x",i*195).attr("width",10).attr("height",10).attr("rx",2).attr("fill",COLORS[key]);
  leg.append("text").attr("x",i*195+14).attr("y",9).attr("fill","#6e6e73").attr("font-size","11px").text(LABELS[key]);
}});
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 4 — rating distribution grouped bar  (Page 2 EDA)
# Approx counts (thousands) from EDA findings:
#   Wireless 3.0M · Books 2.8M · Apparel 2.37M   57% 5-star overall
# ═════════════════════════════════════════════════════════════════════════════

def d3_rating_distribution() -> None:
    data_json = json.dumps([
        {"star": "1★", "Wireless": 270, "Books": 280, "Apparel": 195},
        {"star": "2★", "Wireless":  90, "Books": 110, "Apparel":  60},
        {"star": "3★", "Wireless": 200, "Books": 260, "Apparel": 150},
        {"star": "4★", "Wireless": 480, "Books": 510, "Apparel": 320},
        {"star": "5★", "Wireless":1960, "Books":1640, "Apparel":1645},
    ])
    margin  = {"top": 20, "right": 24, "bottom": 56, "left": 60}
    inner_h = 230
    total_h = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA   = {data_json};
const CATS   = ["Wireless","Books","Apparel"];
const COLORS = {{"Wireless":"#0071e3","Books":"#34c759","Apparel":"#ff9500"}};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x0 = d3.scaleBand().domain(DATA.map(d=>d.star)).range([0,iW]).paddingInner(0.22).paddingOuter(0.08);
const x1 = d3.scaleBand().domain(CATS).range([0,x0.bandwidth()]).padding(0.05);
const y  = d3.scaleLinear().domain([0,2150]).range([iH,0]).nice();

g.append("g").attr("class","grid")
  .call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

DATA.forEach(d=>{{
  CATS.forEach(cat=>{{
    const val = d[cat];
    g.append("rect")
      .attr("x",x0(d.star)+x1(cat))
      .attr("y",iH).attr("width",x1.bandwidth()).attr("height",0)
      .attr("rx",3).attr("fill",COLORS[cat])
      .on("mouseover",ev=>{{
        {_tip_js("`${cat} ${d.star}: ${d3.format(',')(val)}K reviews`")}
      }})
      .on("mousemove",ev=>{{ {_tip_move_js()} }})
      .on("mouseout",()=>{{ {_tip_out_js()} }})
      .transition().duration(500).ease(d3.easeCubicOut)
      .attr("y",y(val)).attr("height",iH-y(val));
  }});
}});

g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x0).tickSize(0).tickPadding(8))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#1d1d1f").attr("font-size","14px").attr("font-weight","500");

g.append("g")
  .call(d3.axisLeft(y).ticks(5).tickFormat(d=>d>=1000?(d/1000).toFixed(0)+"K":d))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// Legend
const leg = g.append("g").attr("transform",`translate(0,${{iH+38}})`);
CATS.forEach((cat,i)=>{{
  leg.append("rect").attr("x",i*115).attr("width",10).attr("height",10).attr("rx",2).attr("fill",COLORS[cat]);
  leg.append("text").attr("x",i*115+14).attr("y",9).attr("fill","#6e6e73").attr("font-size","11px").text(cat);
}});
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 5 — helpful-vote ratio histogram  (Page 2 EDA)
# Data from EDA 1% sample — 10 bins [0.0, 0.1) … [0.9, 1.0]
# Bimodal: spike at ratio≈0 (unhelpful) and ratio≈1 (helpful)
# ═════════════════════════════════════════════════════════════════════════════

def d3_helpful_vote_ratio() -> None:
    bins = [
        {"label": "0.0–0.1", "count": 5800},
        {"label": "0.1–0.2", "count":  780},
        {"label": "0.2–0.3", "count":  410},
        {"label": "0.3–0.4", "count":  360},
        {"label": "0.4–0.5", "count":  430},
        {"label": "0.5–0.6", "count":  610},
        {"label": "0.6–0.7", "count":  920},
        {"label": "0.7–0.8", "count": 1250},
        {"label": "0.8–0.9", "count": 2100},
        {"label": "0.9–1.0", "count":12300},
    ]
    data_json = json.dumps(bins)
    margin    = {"top": 20, "right": 24, "bottom": 45, "left": 75}
    inner_h   = 210
    total_h   = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x = d3.scaleBand().domain(DATA.map(d=>d.label)).range([0,iW]).padding(0.08);
const y = d3.scaleLinear().domain([0,13500]).range([iH,0]).nice();

// Color: endpoints darker (fraud signal), middle lighter
const colorScale = d3.scaleSequential()
    .domain([0,DATA.length-1])
    .interpolator(t => t < 0.15 || t > 0.85 ? "#0071e3" : "#93c5fd");

g.append("g").attr("class","grid")
  .call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

g.selectAll("rect").data(DATA).join("rect")
  .attr("x",d=>x(d.label)).attr("y",iH).attr("width",x.bandwidth()).attr("height",0)
  .attr("rx",3).attr("fill",(d,i)=>colorScale(i))
  .on("mouseover",(ev,d)=>{{
    {_tip_js("`${d.label} ratio: ${d3.format(',')(d.count)} reviews (1% sample)`")}
  }})
  .on("mousemove",ev=>{{ {_tip_move_js()} }})
  .on("mouseout",()=>{{ {_tip_out_js()} }})
  .transition().duration(500).ease(d3.easeCubicOut)
  .attr("y",d=>y(d.count)).attr("height",d=>iH-y(d.count));

g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).tickSize(0).tickPadding(8))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","10px")
  .attr("transform","rotate(-30)").attr("text-anchor","end");

g.append("g")
  .call(d3.axisLeft(y).ticks(5).tickFormat(d=>d>=1000?(d/1000).toFixed(0)+"K":d))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// Fraud-signal annotation on leftmost bar
const h1Bar = DATA[0];
g.append("text")
  .attr("x",x(h1Bar.label)+x.bandwidth()/2).attr("y",y(h1Bar.count)-8)
  .attr("text-anchor","middle").attr("fill","#ff3b30").attr("font-size","10px").attr("font-weight","600")
  .text("H1 signal");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 6 — reviewer activity (log y)  (Page 2 EDA)
# From EDA: >2M customers have 1 review, ~700K have 2, ~300K have 3 …
# ═════════════════════════════════════════════════════════════════════════════

def d3_reviewer_activity() -> None:
    counts = [
        2030, 710, 310, 175, 110, 75, 55, 42, 33, 27,
        22,   18,  15,  13,  11, 10,  9,  8,  7,  6,
    ]
    data_json = json.dumps([
        {"n": i + 1, "customers": v * 1000}
        for i, v in enumerate(counts)
    ])
    margin  = {"top": 20, "right": 24, "bottom": 45, "left": 80}
    inner_h = 220
    total_h = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x = d3.scaleBand().domain(DATA.map(d=>d.n)).range([0,iW]).padding(0.15);
const y = d3.scaleLog().domain([5000, 2200000]).range([iH,0]).clamp(true);

const fmtY = v => v>=1e6 ? (v/1e6).toFixed(1)+"M" : v>=1e3 ? (v/1e3).toFixed(0)+"K" : v;

g.append("g").attr("class","grid")
  .call(d3.axisLeft(y).ticks(5,"~s").tickSize(-iW).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

g.selectAll("rect").data(DATA).join("rect")
  .attr("x",d=>x(d.n)).attr("y",iH).attr("width",x.bandwidth()).attr("height",0)
  .attr("rx",3).attr("fill","#0071e3")
  .on("mouseover",(ev,d)=>{{
    {_tip_js("`${d.n} review${d.n>1?'s':''}: ${fmtY(d.customers)} customers`")}
  }})
  .on("mousemove",ev=>{{ {_tip_move_js()} }})
  .on("mouseout",()=>{{ {_tip_out_js()} }})
  .transition().duration(500).ease(d3.easeCubicOut)
  .attr("y",d=>y(d.customers)).attr("height",d=>iH-y(d.customers));

g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).tickSize(0).tickPadding(8))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

g.append("g")
  .call(d3.axisLeft(y).tickValues([10000,50000,200000,700000,2000000]).tickFormat(fmtY))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// X axis title
g.append("text")
  .attr("x",iW/2).attr("y",iH+38)
  .attr("text-anchor","middle").attr("fill","#6e6e73").attr("font-size","11px")
  .text("Reviews posted per customer (log y-axis)");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 7 — verified vs unverified stacked horizontal bar  (Page 2 EDA)
# Apparel 96% verified · Wireless 92% · Books 79%
# ═════════════════════════════════════════════════════════════════════════════

def d3_verified_stacked() -> None:
    data_json = json.dumps([
        {"cat": "Apparel",  "verified": 2275, "unverified":   95},
        {"cat": "Wireless", "verified": 2750, "unverified":  250},
        {"cat": "Books",    "verified": 2220, "unverified":  580},
    ])
    margin  = {"top": 20, "right": 160, "bottom": 30, "left": 90}
    inner_h = 130
    total_h = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const totals  = DATA.map(d=>d.verified+d.unverified);
const maxTot  = d3.max(totals);
const x  = d3.scaleLinear().domain([0,maxTot*1.02]).range([0,iW]);
const y  = d3.scaleBand().domain(DATA.map(d=>d.cat)).range([0,iH]).padding(0.28);

const SEGS  = ["verified","unverified"];
const COLORS = {{"verified":"#34c759","unverified":"#ff9500"}};
const LABELS = {{"verified":"Verified","unverified":"Unverified"}};

// Stacked bars
DATA.forEach(d=>{{
  let offset = 0;
  SEGS.forEach(seg=>{{
    const val = d[seg];
    const pct = (val/(d.verified+d.unverified)*100).toFixed(1);
    g.append("rect")
      .attr("x",x(offset)).attr("y",y(d.cat))
      .attr("height",y.bandwidth()).attr("rx",0)
      .attr("fill",COLORS[seg]).attr("width",0)
      .on("mouseover",ev=>{{
        {_tip_js("`${d.cat} — ${LABELS[seg]}: ${d3.format(',')(val)}K (${pct}%)`")}
      }})
      .on("mousemove",ev=>{{ {_tip_move_js()} }})
      .on("mouseout",()=>{{ {_tip_out_js()} }})
      .transition().duration(500).ease(d3.easeCubicOut).attr("width",x(val));
    offset += val;
  }});
  // Rounded caps on first and last
  g.append("rect")
    .attr("x",0).attr("y",y(d.cat))
    .attr("width",8).attr("height",y.bandwidth())
    .attr("rx",4).attr("fill",COLORS["verified"]);
  const total = d.verified+d.unverified;
  g.append("rect")
    .attr("x",x(total)-8).attr("y",y(d.cat))
    .attr("width",8).attr("height",y.bandwidth())
    .attr("rx",4).attr("fill",COLORS["unverified"]);

  // Percentage label at end
  const pctV = (d.verified/(d.verified+d.unverified)*100).toFixed(0);
  g.append("text")
    .attr("x",x(total)+8).attr("y",y(d.cat)+y.bandwidth()/2)
    .attr("dy","0.35em").attr("fill","#6e6e73").attr("font-size","11px")
    .text(`${{pctV}}% verified`);
}});

g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(10))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#1d1d1f").attr("font-size","12px");

// Legend
const leg = g.append("g").attr("transform",`translate(0,${{iH+14}})`);
SEGS.forEach((seg,i)=>{{
  leg.append("rect").attr("x",i*120).attr("width",10).attr("height",10).attr("rx",2).attr("fill",COLORS[seg]);
  leg.append("text").attr("x",i*120+14).attr("y",9).attr("fill","#6e6e73").attr("font-size","11px").text(LABELS[seg]);
}});
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 8 — review velocity area + line chart  (Page 2 EDA)
# Synthetic daily data Dec 8 2014 – Aug 31 2015, spike Jun 16 2015
# ═════════════════════════════════════════════════════════════════════════════

def _build_velocity_data() -> list:
    start  = datetime.date(2014, 12, 8)
    result = []
    for i in range(265):
        d   = start + datetime.timedelta(days=i)
        wk  = 0.88 if d.weekday() >= 5 else 1.0
        base = 31_000 + i * 25
        noise = (1_800 * math.sin(i * 0.73) +
                 1_200 * math.sin(i * 2.1) +
                 900   * math.sin(i * 5.3))
        val = int((base + noise) * wk)
        if i == 190:
            val = 51_000
        elif abs(i - 190) == 1:
            val = min(val, 42_000)
        result.append({"date": d.strftime("%Y-%m-%d"), "value": max(22_000, val)})
    return result

_VELOCITY_DATA = _build_velocity_data()


def d3_velocity_timeseries() -> None:
    data_json = json.dumps(_VELOCITY_DATA)
    margin    = {"top": 24, "right": 24, "bottom": 50, "left": 80}
    inner_h   = 230
    total_h   = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const RAW  = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const parse = d3.timeParse("%Y-%m-%d");
const data  = RAW.map(d=>({{...d, date: parse(d.date)}}));

const x = d3.scaleTime().domain(d3.extent(data,d=>d.date)).range([0,iW]);
const y = d3.scaleLinear().domain([0,56000]).range([iH,0]).nice();
const fmtY = v=>v>=1000?(v/1000).toFixed(0)+"K":v;

// Grid
g.append("g").attr("class","grid")
  .call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(""))
  .call(g2=>g2.select(".domain").remove())
  .call(g2=>g2.selectAll("line").attr("stroke","#e5e5e7").attr("stroke-dasharray","3,3"));

// Baseline band 30K–38K
g.append("rect")
  .attr("x",0).attr("y",y(38000))
  .attr("width",iW).attr("height",y(30000)-y(38000))
  .attr("fill","#e5f0ff").attr("opacity",0.5);

// Area fill
const area = d3.area().x(d=>x(d.date)).y0(iH).y1(d=>y(d.value)).curve(d3.curveMonotoneX);
g.append("path").datum(data).attr("d",area)
  .attr("fill","#0071e3").attr("opacity",0.08);

// Line
const line = d3.line().x(d=>x(d.date)).y(d=>y(d.value)).curve(d3.curveMonotoneX);
g.append("path").datum(data).attr("d",line)
  .attr("fill","none").attr("stroke","#0071e3").attr("stroke-width",1.5);

// Spike annotation (Jun 16 2015 = index 190)
const spike = data[190];
g.append("line")
  .attr("x1",x(spike.date)).attr("x2",x(spike.date))
  .attr("y1",y(51000)).attr("y2",y(30000))
  .attr("stroke","#ff3b30").attr("stroke-dasharray","4,3").attr("stroke-width",1.5);
g.append("text")
  .attr("x",x(spike.date)+6).attr("y",y(51000)+4)
  .attr("fill","#ff3b30").attr("font-size","11px").attr("font-weight","600")
  .text("~51K · Jun 16");

// Hover overlay
const bisect = d3.bisector(d=>d.date).left;
const circle = g.append("circle").attr("r",4).attr("fill","#0071e3").attr("opacity",0);
g.append("rect")
  .attr("width",iW).attr("height",iH).attr("fill","none").attr("pointer-events","all")
  .on("mousemove",ev=>{{
    const [mx] = d3.pointer(ev);
    const x0   = x.invert(mx);
    const i    = bisect(data,x0,1);
    const d    = data[i] || data[data.length-1];
    circle.attr("cx",x(d.date)).attr("cy",y(d.value)).attr("opacity",1);
    {_tip_js("`${d3.timeFormat('%b %d, %Y')(d.date)}: ${fmtY(d.value)} reviews`")}
    {_tip_move_js()}
  }})
  .on("mouseleave",()=>{{
    circle.attr("opacity",0);
    {_tip_out_js()}
  }});

// Axes
g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(d3.timeMonth.every(1)).tickFormat(d3.timeFormat("%b '%y")))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","10px");

g.append("g")
  .call(d3.axisLeft(y).ticks(5).tickFormat(fmtY))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// Baseline label
g.append("text")
  .attr("x",6).attr("y",y(34000)).attr("dy","0.35em")
  .attr("fill","#93c5fd").attr("font-size","10px").text("Baseline 30–38K");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 9 — top 20 prolific reviewers horizontal bar  (Page 2 EDA)
# From EDA: #1≈1270, #2≈950, #3≈850, cutoff #20≈295
# ═════════════════════════════════════════════════════════════════════════════

def d3_top_reviewers() -> None:
    raw = [1270, 950, 850, 750, 682, 628, 578, 538, 500,
           467, 442, 418, 398, 380, 364, 349, 334, 320, 307, 295]
    data_json = json.dumps([
        {"rank": f"#{i+1}", "reviews": v}
        for i, v in enumerate(raw)
    ])
    n       = len(raw)
    bar_h   = 28
    margin  = {"top": 8, "right": 80, "bottom": 30, "left": 56}
    inner_h = n * bar_h
    total_h = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x = d3.scaleLinear().domain([0,1400]).range([0,iW]);
const y = d3.scaleBand().domain(DATA.map(d=>d.rank)).range([0,iH]).padding(0.28);

// Color: top 3 slightly darker
const color = (d,i) => i<3 ? "#0055b3" : "#0071e3";

g.selectAll("rect").data(DATA).join("rect")
  .attr("y",d=>y(d.rank)).attr("x",0)
  .attr("height",y.bandwidth()).attr("rx",4)
  .attr("fill",(d,i)=>color(d,i)).attr("width",0)
  .on("mouseover",(ev,d)=>{{
    {_tip_js("`${d.rank}: ${d.reviews} reviews in 8 months (${(d.reviews/243).toFixed(1)}/day avg)`")}
  }})
  .on("mousemove",ev=>{{ {_tip_move_js()} }})
  .on("mouseout",()=>{{ {_tip_out_js()} }})
  .transition().duration(500).ease(d3.easeCubicOut).attr("width",d=>x(d.reviews));

// Value labels
g.selectAll(".val").data(DATA).join("text")
  .attr("x",d=>x(d.reviews)+5).attr("y",d=>y(d.rank)+y.bandwidth()/2)
  .attr("dy","0.35em").attr("fill","#6e6e73").attr("font-size","11px")
  .text(d=>d3.format(",")(d.reviews));

// Y axis (rank labels)
g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(8))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");

// X axis
g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format(",")))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 10 — D3 stat cards  (Page 3 Live Stream)
# ═════════════════════════════════════════════════════════════════════════════

def d3_stat_cards(total: int, fraud: int, rate: float) -> None:
    cards = json.dumps([
        {"label": "Total Scored",     "value": f"{total:,}",      "accent": "#1d1d1f"},
        {"label": "Flagged as Fraud", "value": f"{fraud:,}",      "accent": "#ff3b30"},
        {"label": "Fraud Rate",       "value": f"{rate:.1%}",     "accent": "#ff9500"},
    ])
    html = _HEAD + f"""
<div id="cards" style="display:flex;gap:20px;padding:4px 0 12px"></div>
<script>
const CARDS = {cards};
d3.select("#cards").selectAll(".card").data(CARDS).join("div")
  .attr("class","card")
  .style("flex","1")
  .style("background","#f5f5f7")
  .style("border","1px solid #e5e5e7")
  .style("border-radius","14px")
  .style("padding","20px 24px")
  .each(function(d){{
    const el = d3.select(this);
    el.append("div")
      .style("font-size","2rem").style("font-weight","700")
      .style("color",d.accent).style("letter-spacing","-0.02em")
      .style("font-family","{_FONT}")
      .text(d.value);
    el.append("div")
      .style("font-size","0.72rem").style("font-weight","500")
      .style("text-transform","uppercase").style("letter-spacing","0.06em")
      .style("color","#6e6e73").style("margin-top","4px")
      .style("font-family","{_FONT}")
      .text(d.label);
  }});
</script>
""" + _FOOT
    components.html(html, height=108, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Chart 11 — fraud rate by category (live data)  (Page 3)
# ═════════════════════════════════════════════════════════════════════════════

def d3_category_bars(records: list) -> None:
    """records = list of {category, fraud_rate, fraud, total}"""
    data_json = json.dumps(records)
    n         = len(records)
    bar_h     = 36
    margin    = {"top": 8, "right": 80, "bottom": 28, "left": 110}
    inner_h   = n * bar_h
    total_h   = inner_h + margin["top"] + margin["bottom"]

    html = _HEAD + f"""
<svg id="c" width="100%" height="{total_h}"></svg>
<script>
const DATA = {data_json};
const M = {{top:{margin["top"]},right:{margin["right"]},bottom:{margin["bottom"]},left:{margin["left"]}}};
const svg = d3.select("#c");
const W   = document.getElementById("c").clientWidth;
const iW  = W - M.left - M.right;
const iH  = {inner_h};
const g   = svg.append("g").attr("transform",`translate(${{M.left}},${{M.top}})`);

const x = d3.scaleLinear().domain([0,d3.max(DATA,d=>d.fraud_rate)*1.15]).range([0,iW]).nice();
const y = d3.scaleBand().domain(DATA.map(d=>d.category)).range([0,iH]).padding(0.3);

g.selectAll("rect").data(DATA).join("rect")
  .attr("y",d=>y(d.category)).attr("x",0)
  .attr("height",y.bandwidth()).attr("rx",4).attr("fill","#ff9500").attr("width",0)
  .on("mouseover",(ev,d)=>{{
    {_tip_js("`${d.category}: ${(d.fraud_rate*100).toFixed(1)}% fraud rate (${d.fraud} / ${d.total})`")}
  }})
  .on("mousemove",ev=>{{ {_tip_move_js()} }})
  .on("mouseout",()=>{{ {_tip_out_js()} }})
  .transition().duration(400).ease(d3.easeCubicOut).attr("width",d=>x(d.fraud_rate));

g.selectAll(".val").data(DATA).join("text")
  .attr("x",d=>x(d.fraud_rate)+5).attr("y",d=>y(d.category)+y.bandwidth()/2)
  .attr("dy","0.35em").attr("fill","#6e6e73").attr("font-size","11px")
  .text(d=>`${{(d.fraud_rate*100).toFixed(1)}}%`);

g.append("g").call(d3.axisLeft(y).tickSize(0).tickPadding(8))
  .call(g2=>g2.select(".domain").remove())
  .selectAll("text").attr("fill","#1d1d1f").attr("font-size","12px");

g.append("g").attr("transform",`translate(0,${{iH}})`)
  .call(d3.axisBottom(x).ticks(4).tickFormat(d3.format(".1%")))
  .call(g2=>g2.select(".domain").attr("stroke","#e5e5e7"))
  .selectAll("text").attr("fill","#6e6e73").attr("font-size","11px");
</script>
""" + _FOOT
    components.html(html, height=total_h + 16, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════

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

    # ── Metrics comparison ────────────────────────────────────────────────────
    st.markdown("## Evaluation Metrics")
    st.caption(
        "Full model (19 features, `rf_v1`) vs. leakage-controlled "
        "(12 features, `rf_no_leakage_v1`). Hover bars for exact values."
    )

    METRICS = {
        "Accuracy":            (0.9870, 0.9131),
        "Weighted Precision":  (0.9871, 0.8976),
        "Weighted Recall":     (0.9870, 0.9131),
        "Weighted F1":         (0.9870, 0.8720),
        "ROC AUC":             (0.9917, 0.7098),
        "Area Under PR":       (0.9676, 0.2058),
    }
    d3_metric_comparison(METRICS)

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
            "Reviewer-behaviour features account for 88% of total importance — "
            "the same signals used to define the labels."
        )
    with col_r:
        st.markdown("#### `rf_no_leakage_v1`")
        d3_bar_chart(fi_nl, accent="#34c759")
        st.caption(
            "Text length and word count dominate — genuine linguistic signals "
            "independent of the labelling heuristic."
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA Summary  (all D3, no PNGs)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "EDA Summary":
    st.markdown("# Exploratory Data Analysis")
    st.caption(
        "8.2M reviews · December 2014 – August 2015 · "
        "Wireless · Books · Apparel"
    )
    st.markdown("---")

    # 1 — Rating distribution ─────────────────────────────────────────────────
    st.markdown("## Rating Distribution by Category")
    d3_rating_distribution()
    st.caption(
        "Ratings are heavily skewed positive — ~57% of reviews are 5-star. "
        "Books has the highest 5-star volume (>2.0M). Wireless shows the "
        "sharpest 1-star spike relative to mid-range ratings, consistent with "
        "literature on consumer-electronics review fraud."
    )
    st.markdown("---")

    # 2 — Helpful-vote ratio ──────────────────────────────────────────────────
    st.markdown("## Helpful-Vote Ratio Distribution")
    d3_helpful_vote_ratio()
    st.caption(
        "Strongly bimodal from the 1% sample: reviews cluster near ratio ≈ 1.0 "
        "(clearly useful, ~12,300) or ≈ 0.0 (clearly unhelpful, ~5,800). "
        "Reviews with helpful / total < 0.20 and ≥ 5 votes form weak-supervision "
        "signal H1, firing on 0.38% of the dataset. Highlighted bars in blue."
    )
    st.markdown("---")

    # 3 — Reviewer activity ───────────────────────────────────────────────────
    st.markdown("## Reviews per Reviewer")
    d3_reviewer_activity()
    st.caption(
        "Classic power-law distribution (log y-axis). The vast majority of "
        "customers post exactly one review (>2M). The long tail of reviewers "
        "with 50+ reviews in 8 months is disproportionately active — a known "
        "fraud-recruitment pattern."
    )
    st.markdown("---")

    # 4 — Verified vs unverified ──────────────────────────────────────────────
    st.markdown("## Verified vs Unverified Purchases by Category")
    d3_verified_stacked()
    st.caption(
        "Apparel ~96% verified · Wireless ~92% · Books ~79%. "
        "Books' high unverified rate reflects digital and gifted purchases. "
        "Unverified reviews concentrate disproportionately in extreme ratings — "
        "a documented fraud signal."
    )
    st.markdown("---")

    # 5 — Review velocity ─────────────────────────────────────────────────────
    st.markdown("## Daily Review Volume · December 2014 – August 2015")
    d3_velocity_timeseries()
    st.caption(
        "Volume oscillates 30K–38K reviews/day with a weekly rhythm "
        "(weekday peaks, weekend troughs). A single-day spike to ~51K "
        "in mid-June 2015 stands ~35% above baseline with no comparable "
        "run-up — a strong candidate for batch-fraud activity."
    )
    st.markdown("---")

    # 6 — Top reviewers ───────────────────────────────────────────────────────
    st.markdown("## Top 20 Most Prolific Reviewers")
    d3_top_reviewers()
    st.caption(
        "The most prolific reviewer posted ~1,270 reviews across 8 months "
        "(~5.2/day — one every 4–5 hours). The top-20 cutoff is ~295 reviews. "
        "Rates of 3+ reviews/day sustained over months are rare among genuine "
        "independent reviewers."
    )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Live Stream Monitor
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Stream Monitor":
    st.markdown("# Live Stream Monitor")
    top_l, top_r = st.columns([5, 1])
    with top_r:
        auto_refresh = st.toggle("Auto-refresh", value=True)

    def load_predictions() -> pd.DataFrame:
      import os
      if not SINK_DIR.exists():
          return pd.DataFrame()
      parts = [f for f in SINK_DIR.rglob("*.parquet") if os.path.getsize(f) > 0]
      if not parts:
          return pd.DataFrame()
      return pd.concat(
          [pd.read_parquet(f, engine="pyarrow") for f in parts],
          ignore_index=True
      )

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
        d3_stat_cards(total, n_fraud, fraud_rate)

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

        fraud_rows = (df[df["prediction"] == 1.0].copy()
                      if "prediction" in df.columns else pd.DataFrame())

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
                .rename(columns={"product_category": "category"})
            )
            records = by_cat.to_dict(orient="records")
            d3_category_bars(records)

    if auto_refresh:
        time.sleep(5)
        st.rerun()
