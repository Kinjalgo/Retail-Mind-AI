"""
app.py — AI Customer Analytics Dashboard (FINAL FIXED)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from data_loader import load_dataframes, build_main_dataframe, get_demo_dataframe, get_coupon_dataframe
from analysis_engine import get_kpis, get_spending_trend, get_top_categories
from sql_engine import load_into_sqlite
from ai_agents import get_llm
from ask_question_component import render_ask_question_section
from utils import fmt_currency, fmt_number, fmt_percent

# ---------------- PAGE ----------------
st.set_page_config(page_title="AI Customer Analytics", layout="wide")
st.title("📊 AI Customer Analytics")

# ---------------- SIDEBAR ----------------
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
model_name = st.sidebar.selectbox("Model", ["ollama/qwen2.5-coder:3b"])

# ---------------- LOAD DATA ----------------
if uploaded_files:
    dataframes = load_dataframes(uploaded_files)
    df = build_main_dataframe(dataframes)
    demo_df = get_demo_dataframe(dataframes)
    coupon_df = get_coupon_dataframe(dataframes)
    conn = load_into_sqlite(dataframes)

    kpis = get_kpis(df)

    llm = get_llm(model=model_name, base_url=ollama_url)

    # ---------------- DASHBOARD ----------------
    st.subheader("📊 KPIs")
    col1, col2, col3 = st.columns(3)
    col1.metric("Revenue", fmt_currency(kpis["total_revenue"]))
    col2.metric("Customers", fmt_number(kpis["total_customers"]))
    col3.metric("Avg Basket", fmt_currency(kpis["avg_basket"]))

    # Trend
    trend = get_spending_trend(df)
    st.line_chart(trend.set_index("Week"))

    # Categories
    top_cat = get_top_categories(df)
    st.bar_chart(top_cat.set_index("Category"))

    # ---------------- ASK ----------------
    render_ask_question_section(dataframes, df, demo_df, coupon_df, conn, kpis, llm)

else:
    st.info("Upload data to start")
