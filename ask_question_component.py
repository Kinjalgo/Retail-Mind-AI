"""
ask_question_component.py — Question routing, analysis dispatch, and rendering
"""

import sys
import os
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src not in sys.path:
    sys.path.append(_src)

import anthropic
import streamlit as st
import pandas as pd
from typing import Dict, Optional, Any

from ai_agents import classify_question, generate_sql, generate_insight
from analysis_engine import (
    get_spending_trend,
    get_top_categories,
    get_customer_trends,
    get_behavioral_category_breakdown,
    get_behavioral_full_summary,
    get_all_demographic_analyses,
    get_campaign_analysis,
)
from sql_engine import execute_sql


def render_ask_question_section(
    dataframes: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    demo_df: Optional[pd.DataFrame],
    coupon_df: Optional[pd.DataFrame],
    conn: Any,
    kpis: Dict[str, Any],
    client: anthropic.Anthropic,
):
    st.subheader("💬 Ask a Question")
    q = st.text_input(
        "Enter your question",
        placeholder="e.g. Which categories are growing? Is there evidence direct marketing improves engagement?",
    )
    col_btn, col_sql = st.columns([3, 2])
    with col_btn:
        run = st.button("Analyze", type="primary")
    with col_sql:
        show_sql = st.checkbox("Also generate SQL", value=False,
                               help="Adds ~1,500 tokens (~$0.005) per question")

    if not (run and q):
        return

    q_lower = q.lower()
    q_type = classify_question(q)
    result_df = None
    extra_context = ""

    with st.spinner("Analysing…"):

        # ── BEHAVIORAL ──────────────────────────────────────────────────────
        if any(w in q_lower for w in ["increase", "decrease", "growing", "declining",
                                       "spending more", "spending less",
                                       "spend more", "spend less", "over time",
                                       "how many customers"]):
            q_type = "behavioral"
            direction = (
                "Increasing"
                if any(w in q_lower for w in ["increase", "growing", "spending more"])
                else "Decreasing"
            )

            trend_df = get_customer_trends(df)
            if trend_df.empty:
                st.warning("No customer trend data available.")
                return

            target_customers = trend_df[trend_df["Trend"] == direction]["HOUSEHOLD_KEY"]
            if target_customers.empty:
                st.info(f"No customers found with {direction} spending trend.")
                return

            customer_summary = (
                df[df["HOUSEHOLD_KEY"].isin(target_customers)]
                .groupby("HOUSEHOLD_KEY")["SALES_VALUE"]
                .sum()
                .reset_index()
                .sort_values("SALES_VALUE", ascending=False)
                .head(10)
            )

            category_summary      = get_behavioral_category_breakdown(df, direction)
            behavioral_full_summ  = get_behavioral_full_summary(df)
            result_df             = behavioral_full_summ

            extra_context = (
                f"SUMMARY OF ALL TREND GROUPS:\n{behavioral_full_summ.to_string(index=False)}\n\n"
                f"TOP 10 CUSTOMERS ({direction.upper()}):\n{customer_summary.to_string(index=False)}\n\n"
                f"TOP CATEGORIES FOR {direction.upper()} CUSTOMERS:\n{category_summary.to_string(index=False)}"
            )

            st.subheader("Overall Customer Spending Trends")
            st.dataframe(behavioral_full_summ, use_container_width=True)
            st.subheader(f"Top 10 Customers — {direction} Spend")
            st.dataframe(customer_summary, use_container_width=True)
            st.subheader(f"Top Categories — {direction} Customers")
            st.dataframe(category_summary, use_container_width=True)

        # ── CAMPAIGN / COUPON ────────────────────────────────────────────────
        elif any(w in q_lower for w in ["coupon", "campaign", "promotion", "discount",
                                         "mailer", "marketing", "engagement", "evidence",
                                         "direct market", "improve", "redempt"]):
            q_type = "campaign"
            campaign = get_campaign_analysis(df, coupon_df)

            if campaign:
                comp    = campaign["comparison"]
                avg_pc  = campaign.get("avg_per_customer", pd.DataFrame())
                result_df = comp

                extra_context = (
                    f"GROUP COMPARISON:\n{comp.to_string(index=False)}\n\n"
                    f"AVG SPEND PER CUSTOMER:\n{avg_pc.to_string(index=False)}"
                )
                st.dataframe(comp, use_container_width=True)
            else:
                st.warning("No campaign/coupon data available.")
                return

        # ── DEMOGRAPHIC ──────────────────────────────────────────────────────
        elif any(w in q_lower for w in ["age", "demographic", "income", "marital",
                                         "homeowner", "household", "kids", "family",
                                         "segment", "which factor", "affect", "influence"]):
            q_type = "demographic"
            all_demos = get_all_demographic_analyses(df, demo_df)

            if not all_demos:
                st.warning("No demographic data available.")
                return

            combined = pd.concat(all_demos.values(), ignore_index=True)
            result_df = combined
            extra_context = "\n\n".join(
                f"{dim}:\n{frame.to_string(index=False)}"
                for dim, frame in all_demos.items()
            )
            st.dataframe(combined, use_container_width=True)

        # ── CATEGORY ────────────────────────────────────────────────────────
        elif any(w in q_lower for w in ["category", "product", "commodity", "department", "brand"]):
            q_type = "category"
            result_df = get_top_categories(df)
            st.dataframe(result_df, use_container_width=True)

        # ── TREND (default) ──────────────────────────────────────────────────
        else:
            q_type = "trend"
            result_df = get_spending_trend(df)
            st.line_chart(result_df.set_index("Week"))

        # ── SQL (only when user opts in) ─────────────────────────────────────
        if show_sql:
            try:
                sql = generate_sql(q, dataframes, client)
                sql_result, err = execute_sql(conn, sql)
                with st.expander("🔍 Generated SQL", expanded=True):
                    st.code(sql, language="sql")
                    if sql_result is not None and not sql_result.empty:
                        st.dataframe(sql_result, use_container_width=True)
                    elif err:
                        st.error(f"SQL error: {err}")
            except Exception as e:
                st.warning(f"SQL generation skipped: {e}")

        # ── AI INSIGHT ───────────────────────────────────────────────────────
        if result_df is not None and not result_df.empty:
            try:
                insight = generate_insight(
                    question=q,
                    result_df=result_df,
                    question_type=q_type,
                    client=client,
                    kpi_context=str(kpis),
                    extra_context=extra_context,
                )
                st.subheader("🤖 AI Insight")
                st.markdown(insight)
                st.session_state.questions_asked = (
                    st.session_state.get("questions_asked", 0) + 1
                )
            except Exception as e:
                err_msg = str(e)
                if "401" in err_msg or "authentication" in err_msg.lower():
                    st.error(
                        "**API key invalid.** "
                        "Paste your real key (starts with `sk-ant-`) in the sidebar. "
                        "The data tables above are still correct — only the AI commentary failed."
                    )
                elif "429" in err_msg:
                    st.warning("Rate limit hit. Wait 30 seconds and try again.")
                else:
                    st.warning(f"AI insight unavailable: {e}")
