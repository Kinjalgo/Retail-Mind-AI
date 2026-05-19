import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import sqlite3
from typing import Dict, Optional, Any
from crewai import LLM

# Import necessary functions from other modules
from ai_agents import classify_question, generate_sql, generate_local_insight
from analysis_engine import (
    get_customer_trends,
    get_category_growth_decline,
    get_direct_marketing_effect,
    get_customer_engagement
)
from sql_engine import execute_sql

def render_auto_visualization(result_df):

    import plotly.express as px

    if result_df is None or result_df.empty:
        return

    try:

        numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = result_df.select_dtypes(exclude=["number"]).columns.tolist()

        # ----------------------------------------
        # Trend Line
        # ----------------------------------------

        possible_time = [
            c for c in result_df.columns
            if "week" in c.lower()
            or "month" in c.lower()
            or "date" in c.lower()
        ]

        if possible_time and numeric_cols:

            fig = px.line(
                result_df,
                x=possible_time[0],
                y=numeric_cols[0],
                title="Trend Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)
            return

        # ----------------------------------------
        # Category Comparison
        # ----------------------------------------

        if categorical_cols and numeric_cols:

            fig = px.bar(
                result_df.head(15),
                x=categorical_cols[0],
                y=numeric_cols[0],
                color=categorical_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}"
            )

            st.plotly_chart(fig, use_container_width=True)
            return

        # ----------------------------------------
        # Correlation / Scatter
        # ----------------------------------------

        if len(numeric_cols) >= 2:

            fig = px.scatter(
                result_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                size=numeric_cols[-1] if len(numeric_cols) >= 3 else None,
                title="Relationship Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)
            return

    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")

def render_ask_question_section(
    dataframes: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    demo_df: Optional[pd.DataFrame],
    coupon_df: Optional[pd.DataFrame],
    conn: sqlite3.Connection,
    kpis: Dict[str, Any],
    llm: LLM
):
    """
    Renders the "Ask a Question" section of the dashboard and handles user queries.

    Args:
        dataframes: A dictionary of all loaded DataFrames.
        df: The main merged DataFrame.
        demo_df: The demographic DataFrame, if available.
        coupon_df: The coupon DataFrame, if available.
        conn: The SQLite database connection.
        kpis: Dictionary of Key Performance Indicators.
        llm: The language model instance.
    """
    st.subheader("💬 Ask a Question")
    q = st.text_input(
    "",
    placeholder="Ask a question...",
    label_visibility="collapsed"
)
    analyze_clicked = st.button(    
        "Analyze Question", use_container_width=True    
        )

    use_ai_narrative = True
   
    
    # =========================
    # ANALYZE QUESTION
    # =========================

    if analyze_clicked:

        q_lower = q.lower()

        # -------------------------
        # SAFETY DEFAULTS
        # -------------------------

        result_df = pd.DataFrame()
        local_insight = ""
        extra_context = ""

        # -------------------------
        # QUESTION CLASSIFICATION
        # -------------------------

        q_type = classify_question(q)

        # =========================
        # KPI
        # =========================

        if q_type == "kpi":

            st.subheader("📊 KPI Analysis")

            result_df = pd.DataFrame([kpis])

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.markdown(local_insight)

        # =========================
        # BEHAVIORAL
        # =========================

        elif q_type == "behavioral":

            st.subheader("📈 Behavioral Analysis")

            result_df = get_customer_trends(df)

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.markdown(local_insight)

            render_auto_visualization(result_df)

        # =========================
        # DEMOGRAPHIC
        # =========================

        elif q_type == "demographic":

            st.subheader("👥 Demographic Intelligence")

            result_df = get_customer_engagement(df, demo_df)

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.markdown(local_insight)

            render_auto_visualization(result_df)

        # =========================
        # CATEGORY
        # =========================

        elif q_type == "category_growth":

            st.subheader("🛒 Category Analysis")

            result_df = get_category_growth_decline(df)

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.markdown(local_insight)

            render_auto_visualization(result_df)

        # =========================
        # DIRECT MARKETING
        # =========================

        elif q_type in ["campaign", "direct_marketing"]:

            st.subheader("📬 Direct Marketing Impact")

            result_df = get_direct_marketing_effect(
                df,
                dataframes
            )

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.markdown(local_insight)

            render_auto_visualization(result_df)

        # =========================
        # FALLBACK
        # =========================

        else:

            st.warning("Question type not recognized.")

        # =========================
        # LOCAL INSIGHT
        # =========================

    if analyze_clicked:

        q_lower = q.lower()

        result_df = pd.DataFrame()
        local_insight = ""

        q_type = classify_question(q)

        # KPI
        if q_type == "kpi":

            result_df = pd.DataFrame([kpis])

        # Behavioral
        elif q_type == "behavioral":

            result_df = get_customer_trends(df)

        # Demographic
        elif q_type == "demographic":

            result_df = get_customer_engagement(df, demo_df)

        # Category
        elif q_type == "category_growth":

            result_df = get_category_growth_decline(df)

        # Marketing
        elif q_type in ["campaign", "direct_marketing"]:

            result_df = get_direct_marketing_effect(df, dataframes)

        else:

            st.warning("Question type not recognized.")

        # ALWAYS INSIDE analyze_clicked
        if result_df is not None and not result_df.empty:

            st.dataframe(result_df, use_container_width=True)

            local_insight = generate_local_insight(
                q,
                result_df,
                q_type
            )

            st.subheader("Insight")
            st.markdown(local_insight)

            render_auto_visualization(result_df)

            # OPTIONAL AI INTERPRETATION
            try:

                st.subheader("AI Business Interpretation")
                st.markdown(ai_insight)

            except Exception as e:

                st.warning(f"AI interpretation unavailable: {e}")