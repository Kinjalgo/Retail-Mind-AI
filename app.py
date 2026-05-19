"""
app.py — AI Retail Analytics Dashboard (Claude API + local data edition)
"""

import sys
import os
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.append(_src)

import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

from data_loader import (
    load_from_directory,
    load_causal_data,
    build_main_dataframe,
    get_demo_dataframe,
    get_coupon_dataframe,
)
from analysis_engine import get_kpis, get_spending_trend, get_top_categories
from sql_engine import load_into_sqlite
from ai_agents import get_client
from ask_question_component import render_ask_question_section
from utils import fmt_currency, fmt_number

load_dotenv()

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Retail Analytics", layout="wide")
st.title("🛒 AI Retail Analytics — Dunnhumby Complete Journey")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    # API key — reads from .env, but only trusts it if it looks like a real key
    env_key = os.getenv("ANTHROPIC_API_KEY", "")
    if env_key.startswith("sk-ant-"):
        api_key = env_key
        st.success("API key loaded from .env")
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            placeholder="sk-ant-api03-...",
            help="Paste your key from console.anthropic.com → API Keys",
        )
        if env_key and not env_key.startswith("sk-ant-"):
            st.warning(".env has a placeholder — paste your real key above")

    st.divider()
    st.subheader("📂 Data Source")

    # Resolve data/ folder relative to this script
    default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    data_dir = st.text_input("Data folder path", value=default_data_dir)

    load_causal = st.checkbox(
        "Load causal_data.csv (664 MB — display & mailer analysis)",
        value=False,
        help="Only tick this if you want to analyse in-store promotions or mailer effects. "
             "Adds ~2 min loading time.",
    )

    st.divider()
    st.subheader("💰 Session Cost Tracker")
    if "questions_asked" not in st.session_state:
        st.session_state.questions_asked = 0
    cost_per_q    = 0.023   # insight only
    cost_with_sql = 0.028   # insight + SQL
    total_est = st.session_state.questions_asked * cost_per_q
    st.metric("Questions asked", st.session_state.questions_asked)
    st.metric("Est. session cost", f"${total_est:.3f}")
    st.caption("~$0.023 per question  |  ~$0.028 with SQL\n\n"
               "Files loaded automatically from the data folder. "
               "causal_data.csv excluded by default (664 MB).")

# ── GATE: API key required ─────────────────────────────────────────────────────
if not api_key:
    st.warning("Add your Anthropic API key in the sidebar to enable AI analysis.")
    st.stop()

# ── LOAD DATA FROM DISK ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data from disk…")
def _load(data_dir: str, include_causal: bool):
    dataframes = load_from_directory(data_dir)
    if include_causal:
        causal = load_causal_data(data_dir)
        if causal is not None:
            dataframes["causal_data.csv"] = causal
    return dataframes

dataframes = _load(data_dir, load_causal)

if not dataframes:
    st.error(f"No CSV files found in: {data_dir}\nCheck the path in the sidebar.")
    st.stop()

df        = build_main_dataframe(dataframes)
demo_df   = get_demo_dataframe(dataframes)
coupon_df = get_coupon_dataframe(dataframes)

if df is None or df.empty:
    st.error("Could not build main dataframe — transaction_data.csv may be missing.")
    st.stop()

conn   = load_into_sqlite(dataframes)
kpis   = get_kpis(df)
client = get_client(api_key)

# ── DATA SUMMARY BANNER ────────────────────────────────────────────────────────
with st.expander("📋 Loaded files", expanded=False):
    rows = [
        {"File": k, "Rows": f"{len(v):,}", "Columns": len(v.columns)}
        for k, v in dataframes.items()
    ]
    st.table(pd.DataFrame(rows))

# ── KPIs ───────────────────────────────────────────────────────────────────────
st.subheader("📊 Key Performance Indicators")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue",       fmt_currency(kpis["total_revenue"]))
c2.metric("Total Customers",     fmt_number(kpis["total_customers"]))
c3.metric("Total Transactions",  fmt_number(kpis["total_transactions"]))
c4.metric("Avg Basket Size",     fmt_currency(kpis["avg_basket"]))
c5.metric("Revenue / Customer",  fmt_currency(kpis["revenue_per_customer"]))

st.divider()

# ── CHARTS ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📈 Weekly Revenue Trend")
    trend = get_spending_trend(df)
    fig = px.line(
        trend, x="Week", y="Revenue", markers=True,
        labels={"Revenue": "Revenue ($)", "Week": "Week No."},
    )
    fig.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🏷️ Top 10 Categories by Revenue")
    top_cat = get_top_categories(df)
    fig2 = px.bar(
        top_cat, x="Revenue", y="Category", orientation="h",
        text_auto=".2s",
        labels={"Revenue": "Revenue ($)", "Category": ""},
    )
    fig2.update_layout(margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── ASK A QUESTION ─────────────────────────────────────────────────────────────
render_ask_question_section(dataframes, df, demo_df, coupon_df, conn, kpis, client)
