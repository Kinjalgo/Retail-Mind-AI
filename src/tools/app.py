"""
app.py — RetailMind AI · Executive Analytics Dashboard
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_loader import load_dataframes, build_main_dataframe, get_demo_dataframe, get_coupon_dataframe
from analysis_engine import get_kpis, get_spending_trend, get_top_categories
from sql_engine import load_into_sqlite
from ai_agents import get_llm
from ask_question_component import render_ask_question_section
from utils import fmt_currency, fmt_number, fmt_percent

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RetailMind AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background: #080B14 !important;
    color: #E2E8F0 !important;
}
.stApp { background: #080B14; }

/* ── Block container ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1440px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #1E2D45 !important;
}
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] .sidebar-section-label {
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ── Hero header ── */
.hero-wrap {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1E2D45;
    margin-bottom: 2.5rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}
.hero-left {}
.hero-wordmark {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    color: #F8FAFC;
    margin-bottom: 0.4rem;
}
.hero-wordmark span {
    color: #3B82F6;
}
.hero-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 300;
    color: #475569;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.hero-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #3B82F6;
    border: 1px solid #1E3A5F;
    background: #0F1E35;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    letter-spacing: 1px;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #475569;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1E2D45;
}

/* ── KPI cards ── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    background: #1E2D45;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.kpi-cell {
    background: #0D1117;
    padding: 1.5rem 1.75rem;
    position: relative;
}
.kpi-cell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi-cell.blue::before  { background: #3B82F6; }
.kpi-cell.teal::before  { background: #14B8A6; }
.kpi-cell.amber::before { background: #F59E0B; }
.kpi-cell.rose::before  { background: #F43F5E; }
.kpi-cell.violet::before{ background: #8B5CF6; }

.kpi-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #475569;
    margin-bottom: 0.6rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F1F5F9;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.kpi-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #334155;
}
.kpi-delta-pos { color: #14B8A6; }
.kpi-delta-neg { color: #F43F5E; }

/* ── Chart cards ── */
.chart-card {
    background: #0D1117;
    border: 1px solid #1E2D45;
    border-radius: 12px;
    padding: 1.5rem 1.5rem 0.5rem;
    margin-bottom: 1.5rem;
}
.chart-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #64748B;
    margin-bottom: 1rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1E2D45 !important;
    gap: 0 !important;
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #F1F5F9 !important;
    border-bottom-color: #3B82F6 !important;
}

/* ── Inputs ── */
.stTextInput input,
.stTextArea textarea,
.stSelectbox select {
    background: #0D1117 !important;
    border: 1px solid #1E2D45 !important;
    color: #E2E8F0 !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
}
.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #3B82F6 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #2563EB !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(59,130,246,0.3) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0D1117 !important;
    border: 1px dashed #1E2D45 !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Metric override ── */
[data-testid="stMetric"] { display: none; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe {
    border-radius: 8px !important;
}

/* ── Alert / insight box ── */
.insight-box {
    background: linear-gradient(135deg, #0D1117, #0F1E35);
    border: 1px solid #1E3A5F;
    border-left: 3px solid #3B82F6;
    border-radius: 10px;
    padding: 1.5rem 1.75rem;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    line-height: 1.75;
    color: #CBD5E1;
    white-space: pre-wrap;
    margin-top: 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080B14; }
::-webkit-scrollbar-thumb { background: #1E2D45; border-radius: 2px; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0D1117 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 10px !important;
}

/* ── Code ── */
code, pre {
    font-family: 'JetBrains Mono', monospace !important;
    background: #050810 !important;
    border: 1px solid #1E2D45 !important;
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}

/* ── Upload success pill ── */
.upload-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #0A2A1A;
    border: 1px solid #166534;
    color: #4ADE80;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    margin: 0.2rem 0;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 6rem 2rem;
    color: #1E2D45;
}
.empty-state-icon {
    font-size: 4rem;
    margin-bottom: 1.5rem;
    opacity: 0.4;
}
.empty-state-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1E3A5F;
    margin-bottom: 0.75rem;
}
.empty-state-body {
    font-size: 0.85rem;
    color: #1E2D45;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────

PALETTE = ["#3B82F6","#14B8A6","#F59E0B","#F43F5E","#8B5CF6",
           "#06B6D4","#84CC16","#FB923C","#EC4899","#A78BFA"]

def styled_fig(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(family="Inter, sans-serif", color="#64748B", size=11),
        margin=dict(l=0, r=0, t=36, b=0),
        colorway=PALETTE,
        title_font=dict(family="Syne, sans-serif", size=13, color="#64748B"),
        legend=dict(
            bgcolor="#0D1117",
            bordercolor="#1E2D45",
            borderwidth=1,
            font=dict(size=11, color="#64748B"),
        ),
        xaxis=dict(
            gridcolor="#111827",
            linecolor="#1E2D45",
            tickfont=dict(size=10, color="#475569"),
            title_font=dict(color="#334155"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="#111827",
            linecolor="#1E2D45",
            tickfont=dict(size=10, color="#475569"),
            title_font=dict(color="#334155"),
            zeroline=False,
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# HERO HEADER (always visible)
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-wrap">
  <div class="hero-left">
    <div class="hero-wordmark">Retail<span>Mind</span></div>
    <div class="hero-tagline">Customer Intelligence · Engagement Analytics · Campaign Insights</div>
  </div>
  <div class="hero-badge">◈ AI-POWERED · CREWAI</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:3px;color:#475569;margin-bottom:1.25rem;">Data Sources</p>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Retail CSVs",
        type=["csv"],
        accept_multiple_files=True,
        help="transaction_data.csv · product.csv · hh_demographic.csv · coupon_redempt.csv",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for f in uploaded_files:
            st.markdown(f'<div class="upload-pill">✓ {f.name}</div>', unsafe_allow_html=True)

    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:3px;color:#475569;margin:1.75rem 0 1rem;">AI Engine</p>', unsafe_allow_html=True)

    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434", label_visibility="collapsed",
        placeholder="Ollama URL — http://localhost:11434")

    model_name = st.selectbox("Model", [
        "ollama/llama3.1",
        "ollama/qwen2.5-coder:3b",
        "ollama/gemma3:4b",
    ], label_visibility="collapsed")

    tavily_api_key = st.text_input(
        "Tavily API Key",
        type="password",
        placeholder="Tavily API Key (optional)",
        label_visibility="collapsed",
    )
    if tavily_api_key:
        os.environ["TAVILY_API_KEY"] = tavily_api_key

    st.markdown('<p style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#1E2D45;margin-top:2rem;line-height:1.8;">Expected files<br>transaction_data.csv<br>product.csv<br>hh_demographic.csv<br>coupon_redempt.csv</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING  (cached)
# ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Processing data…")
def process_data(uploaded_files):
    dataframes = load_dataframes(uploaded_files)
    df = build_main_dataframe(dataframes)
    demo_df = get_demo_dataframe(dataframes)
    coupon_df = get_coupon_dataframe(dataframes)
    if df is None or "SALES_VALUE" not in df.columns:
        return dataframes, df, demo_df, coupon_df, None
    kpis = get_kpis(df)
    return dataframes, df, demo_df, coupon_df, kpis


@st.cache_resource(show_spinner="Initialising database…")
def setup_database(_dataframes):
    return load_into_sqlite(_dataframes)


@st.cache_resource(show_spinner="Starting AI engine…")
def setup_llm(model, base_url):
    return get_llm(model=model, base_url=base_url)


# ─────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────

if not uploaded_files:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-state-icon">◈</div>
      <div class="empty-state-title">No data loaded</div>
      <div class="empty-state-body">
        Upload your retail CSV files from the sidebar to begin.<br><br>
        <strong style="color:#1E3A5F;">transaction_data.csv</strong> &nbsp;·&nbsp;
        <strong style="color:#1E3A5F;">product.csv</strong> &nbsp;·&nbsp;
        <strong style="color:#1E3A5F;">hh_demographic.csv</strong> &nbsp;·&nbsp;
        <strong style="color:#1E3A5F;">coupon_redempt.csv</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────
# LOAD + VALIDATE
# ─────────────────────────────────────────────────────────────────

dataframes, df, demo_df, coupon_df, kpis = process_data(uploaded_files)

if kpis is None:
    st.error("SALES_VALUE column missing. Please upload transaction_data.csv.")
    st.stop()

conn = setup_database(dataframes)
llm  = setup_llm(model_name, ollama_url)


# ─────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Key Metrics</div>', unsafe_allow_html=True)

wow = kpis.get("wow_change")
wow_html = ""
if wow is not None:
    cls = "kpi-delta-pos" if wow >= 0 else "kpi-delta-neg"
    arrow = "▲" if wow >= 0 else "▼"
    wow_html = f'<div class="kpi-sub"><span class="{cls}">{arrow} {abs(wow):.1f}% WoW</span></div>'

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-cell blue">
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-value">{fmt_currency(kpis['total_revenue'])}</div>
    {wow_html}
  </div>
  <div class="kpi-cell teal">
    <div class="kpi-label">Customers</div>
    <div class="kpi-value">{fmt_number(kpis['total_customers'])}</div>
    <div class="kpi-sub" style="color:#334155;">unique households</div>
  </div>
  <div class="kpi-cell amber">
    <div class="kpi-label">Transactions</div>
    <div class="kpi-value">{fmt_number(kpis['total_transactions'])}</div>
    <div class="kpi-sub" style="color:#334155;">total records</div>
  </div>
  <div class="kpi-cell rose">
    <div class="kpi-label">Avg Basket</div>
    <div class="kpi-value">{fmt_currency(kpis['avg_basket'])}</div>
    <div class="kpi-sub" style="color:#334155;">per transaction</div>
  </div>
  <div class="kpi-cell violet">
    <div class="kpi-label">Rev / Customer</div>
    <div class="kpi-value">{fmt_currency(kpis['revenue_per_customer'])}</div>
    <div class="kpi-sub" style="color:#334155;">lifetime avg</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# CHARTS ROW
# ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Performance Overview</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([3, 2], gap="medium")

with col_left:
    trend = get_spending_trend(df)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend["Week"],
        y=trend["Revenue"],
        mode="lines",
        line=dict(color="#3B82F6", width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.06)",
        name="Revenue",
        hovertemplate="Week %{x}<br>$%{y:,.0f}<extra></extra>",
    ))
    fig_trend.update_layout(
        title="Weekly Revenue Trend",
        showlegend=False,
    )
    st.plotly_chart(styled_fig(fig_trend, height=340), use_container_width=True)

with col_right:
    top_cat = get_top_categories(df, n=10)

    fig_cat = px.bar(
        top_cat.sort_values("Revenue"),
        x="Revenue",
        y="Category",
        orientation="h",
        color="Revenue",
        color_continuous_scale=["#0F1E35", "#3B82F6"],
        text=top_cat.sort_values("Revenue")["Revenue"].apply(lambda v: f"${v/1000:.0f}K"),
    )
    fig_cat.update_traces(textposition="outside", textfont=dict(size=10, color="#475569"))
    fig_cat.update_coloraxes(showscale=False)
    fig_cat.update_layout(title="Top Categories by Revenue")
    st.plotly_chart(styled_fig(fig_cat, height=340), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
# ASK ANYTHING SECTION
# ─────────────────────────────────────────────────────────────────

st.markdown('<div class="section-label" style="margin-top:1rem;">Intelligence Layer</div>', unsafe_allow_html=True)

render_ask_question_section(
    dataframes,
    df,
    demo_df,
    coupon_df,
    conn,
    kpis,
    llm,
)