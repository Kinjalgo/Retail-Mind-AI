"""
analysis_engine.py — Pure analytics functions (pandas-based, no LLM)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from typing import Optional, Dict, Any, List


# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────

def get_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    total_revenue = df["SALES_VALUE"].sum()
    total_transactions = len(df)
    total_customers = df["HOUSEHOLD_KEY"].nunique()
    avg_basket = total_revenue / max(total_transactions, 1)
    revenue_per_customer = total_revenue / max(total_customers, 1)

    wow_change = None
    if "WEEK_NO" in df.columns:
        weekly = df.groupby("WEEK_NO")["SALES_VALUE"].sum().sort_index()
        if len(weekly) >= 2:
            wow_change = ((weekly.iloc[-1] - weekly.iloc[-2]) / max(weekly.iloc[-2], 1)) * 100

    return {
        "total_revenue": total_revenue,
        "total_transactions": total_transactions,
        "total_customers": total_customers,
        "avg_basket": avg_basket,
        "revenue_per_customer": revenue_per_customer,
        "wow_change": wow_change,
    }


# ─────────────────────────────────────────────
# SPENDING TREND
# ─────────────────────────────────────────────

def get_spending_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("WEEK_NO")["SALES_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"WEEK_NO": "Week", "SALES_VALUE": "Revenue"})
        .sort_values("Week")
    )


# ─────────────────────────────────────────────
# TOP CATEGORIES
# ─────────────────────────────────────────────

def get_top_categories(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (
        df.groupby("COMMODITY_DESC")["SALES_VALUE"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "Revenue", "count": "Transactions"})
        .sort_values("Revenue", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"COMMODITY_DESC": "Category"})
    )


# ─────────────────────────────────────────────
# CUSTOMER SEGMENTATION
# ─────────────────────────────────────────────

def get_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    customer_spend = df.groupby("HOUSEHOLD_KEY")["SALES_VALUE"].sum()
    q33 = customer_spend.quantile(0.33)
    q66 = customer_spend.quantile(0.66)

    def segment(x):
        if x <= q33:
            return "Low"
        elif x <= q66:
            return "Mid"
        else:
            return "High"

    seg_series = customer_spend.apply(segment)
    counts = seg_series.value_counts().reset_index()
    counts.columns = ["Segment", "Count"]

    revenue = (
        pd.DataFrame({"HOUSEHOLD_KEY": customer_spend.index, "SALES_VALUE": customer_spend.values})
        .assign(Segment=seg_series.values)
        .groupby("Segment")["SALES_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"SALES_VALUE": "Revenue"})
    )

    return counts.merge(revenue, on="Segment")


# ─────────────────────────────────────────────
# CUSTOMER FEATURES
# ─────────────────────────────────────────────

def build_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build customer-level behavioral features.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    features = (
        df.groupby("HOUSEHOLD_KEY")
        .agg(
            Total_Spend=("SALES_VALUE", "sum"),
            Transactions=("SALES_VALUE", "count"),
            Avg_Basket=("SALES_VALUE", "mean"),
            Category_Diversity=("COMMODITY_DESC", "nunique"),
            Weeks_Active=("WEEK_NO", "nunique")
        )
        .reset_index()
    )

    features["Frequency"] = features["Transactions"]

    features["Engagement_Score"] = (
        (
            features["Frequency"] / features["Frequency"].max()
        ) * 0.4
        +
        (
            features["Category_Diversity"] / features["Category_Diversity"].max()
        ) * 0.3
        +
        (
            features["Weeks_Active"] / features["Weeks_Active"].max()
        ) * 0.3
    ) * 100

    features["Engagement_Score"] = features["Engagement_Score"].round(2)

    return features


def get_behavioral_category_breakdown(df: pd.DataFrame, direction: str = "Increasing") -> pd.DataFrame:
    """
    For customers trending in the given direction, return their top categories
    with revenue, transactions, unique customers, avg spend, and revenue share.
    Data is sorted by Revenue descending so row 0 is always the #1 category.
    """
    trend_df = get_customer_trends(df)
    if trend_df.empty:
        return pd.DataFrame()

    target_customers = trend_df[trend_df["Trend"] == direction]["HOUSEHOLD_KEY"]
    n_customers = len(target_customers)
    if n_customers == 0:
        return pd.DataFrame()

    filtered = df[df["HOUSEHOLD_KEY"].isin(target_customers)]

    result = (
        filtered.groupby("COMMODITY_DESC")
        .agg(
            Revenue=("SALES_VALUE", "sum"),
            Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
        )
        .sort_values("Revenue", ascending=False)   # highest revenue FIRST
        .head(15)
        .reset_index()
        .rename(columns={"COMMODITY_DESC": "Category"})
    )

    result["Avg_Spend_Per_Customer"] = (result["Revenue"] / result["Unique_Customers"]).round(2)
    result["Revenue_Share_%"] = (result["Revenue"] / result["Revenue"].sum() * 100).round(1)
    result["Rank"] = range(1, len(result) + 1)

    return result


def get_behavioral_full_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary: count + revenue for each trend group.
    Includes avg change % per group.
    """
    trend_df = get_customer_trends(df)
    if trend_df.empty:
        return pd.DataFrame()

    merged = df.merge(trend_df[["HOUSEHOLD_KEY", "Trend", "Change_Pct"]], on="HOUSEHOLD_KEY", how="left")

    summary = (
        merged.groupby("Trend")
        .agg(
            Total_Revenue=("SALES_VALUE", "sum"),
            Total_Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
            Avg_Change_Pct=("Change_Pct", "mean"),
        )
        .reset_index()
    )
    summary["Avg_Revenue_Per_Customer"] = (
        summary["Total_Revenue"] / summary["Unique_Customers"]
    ).round(2)
    return summary


def get_demographic_profile(
    demo_df: Optional[pd.DataFrame], 
    household_keys: pd.Series, 
) -> Dict[str, pd.DataFrame]:
    """
    For a given list of household keys, generate a demographic profile by
    counting occurrences in each available demographic dimension.
    Returns a dictionary of DataFrames, one for each dimension.
    """
    if demo_df is None or household_keys.empty:
        return {}
    
    available_dims = get_available_demo_dimensions(demo_df)
    if not available_dims:
        return {}

    segment_demo_df = demo_df[demo_df["HOUSEHOLD_KEY"].isin(household_keys)]
    if segment_demo_df.empty:
        return {}

    profiles = {}
    for dim in available_dims:
        if dim in segment_demo_df.columns:
            profile = segment_demo_df[dim].value_counts(normalize=True).mul(100).round(1).reset_index()
            profile.columns = [dim, "Percentage"]
            profiles[dim] = profile
    return profiles

# ─────────────────────────────────────────────
# DEMOGRAPHIC ANALYSIS  (all dimensions)
# ─────────────────────────────────────────────

DEMO_DIMENSIONS: List[str] = [
    "AGE_DESC",
    "INCOME_DESC",
    "MARITAL_STATUS_CODE",
    "HOMEOWNER_DESC",
    "HH_COMP_DESC",
    "HOUSEHOLD_SIZE_DESC",
    "KID_CATEGORY_DESC",
]


def get_demographic_analysis(
    df: pd.DataFrame,
    demo_df: Optional[pd.DataFrame],
    dimension: str = "AGE_DESC",
) -> pd.DataFrame:
    """
    Merge transactions with demographics and group by any dimension column.
    Falls back to the first available dimension if the requested one is missing.
    """
    if demo_df is None or "HOUSEHOLD_KEY" not in demo_df.columns:
        return pd.DataFrame()

    merged = df.merge(demo_df, on="HOUSEHOLD_KEY", how="left")

    if dimension not in merged.columns:
        for dim in DEMO_DIMENSIONS:
            if dim in merged.columns:
                dimension = dim
                break
        else:
            return pd.DataFrame()

    result = (
        merged.groupby(dimension)
        .agg(
            Total_Revenue=("SALES_VALUE", "sum"),
            Avg_Spend=("SALES_VALUE", "mean"),
            Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
        )
        .sort_values("Total_Revenue", ascending=False)
        .reset_index()
        .rename(columns={dimension: "Group"})
    )

    result["Revenue_Share_%"] = (
        result["Total_Revenue"] / result["Total_Revenue"].sum() * 100
    ).round(1)
    result["Avg_Revenue_Per_Customer"] = (
        result["Total_Revenue"] / result["Unique_Customers"]
    ).round(2)
    result.attrs["dimension"] = dimension
    return result


def get_all_demographic_analyses(
    df: pd.DataFrame,
    demo_df: Optional[pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Run analysis for every available demographic dimension."""
    if demo_df is None:
        return {}
    results = {}
    for dim in DEMO_DIMENSIONS:
        if dim in demo_df.columns:
            r = get_demographic_analysis(df, demo_df, dimension=dim)
            if not r.empty:
                results[dim] = r
    return results


def get_top_categories_for_demographic_segment(
    df: pd.DataFrame,
    demo_df: Optional[pd.DataFrame],
    dimension: str,
    group_value: Any
) -> pd.DataFrame:
    """
    For a given demographic segment (e.g., dimension='AGE_DESC', group_value='25-34'),
    find their top 10 purchased categories by revenue.
    """
    if demo_df is None or dimension not in demo_df.columns or group_value is None:
        return pd.DataFrame()

    # Find households belonging to the target segment
    target_households = demo_df[demo_df[dimension] == group_value]["HOUSEHOLD_KEY"]
    if target_households.empty:
        return pd.DataFrame()

    # Filter main transaction df for these households
    segment_df = df[df["HOUSEHOLD_KEY"].isin(target_households)]
    
    if segment_df.empty:
        return pd.DataFrame()

    # Use existing get_top_categories function
    return get_top_categories(segment_df, n=10)

def get_available_demo_dimensions(demo_df: Optional[pd.DataFrame]) -> List[str]:
    if demo_df is None:
        return []
    return [d for d in DEMO_DIMENSIONS if d in demo_df.columns]


def get_marketing_attribution(df: pd.DataFrame, coupon_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Analyzes attribution: Did they buy because of a Coupon, Campaign, or 'Anyway' (Organic)?
    """
    if coupon_df is None or "HOUSEHOLD_KEY" not in coupon_df.columns:
        return pd.DataFrame({
            "Attribution": ["Organic (Anyway)"], 
            "Total_Revenue": [df["SALES_VALUE"].sum()],
            "Transactions": [len(df)],
            "Unique_Customers": [df["HOUSEHOLD_KEY"].nunique() if "HOUSEHOLD_KEY" in df.columns else 0],
            "Revenue_Share_%": [100.0]
        })

    # Simple attribution: If household is in coupon_df, mark as 'Marketing Influenced'
    # In a real scenario, you'd match by date and product_id
    coupon_households = set(coupon_df["HOUSEHOLD_KEY"].unique())
    
    df = df.copy()
    df["Attribution"] = df["HOUSEHOLD_KEY"].apply(
        lambda x: "Marketing Influenced" if x in coupon_households else "Organic (Anyway)"
    )
    
    # If you have 'DESCRIPTION' or 'CHANNEL' in coupon_df/campaign data, you can split further
    # Example placeholder for 'Display' logic:
    if "DESCRIPTION" in coupon_df.columns:
        display_hh = set(coupon_df[coupon_df["DESCRIPTION"].str.contains("Display", na=False, case=False)]["HOUSEHOLD_KEY"])
        df.loc[df["HOUSEHOLD_KEY"].isin(display_hh), "Attribution"] = "Display"

    attribution = (
        df.groupby("Attribution")
        .agg(
            Total_Revenue=("SALES_VALUE", "sum"),
            Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique")
        )
        .reset_index()
    )
    
    attribution["Revenue_Share_%"] = (
        attribution["Total_Revenue"] / attribution["Total_Revenue"].sum() * 100
    ).round(1)
    
    return attribution.sort_values("Total_Revenue", ascending=False)


# ─────────────────────────────────────────────
# CAMPAIGN / COUPON ANALYSIS  (fully expanded)
# ─────────────────────────────────────────────

def get_campaign_analysis(
    df: pd.DataFrame,
    coupon_df: Optional[pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Full coupon impact analysis:
    - Overall comparison (Coupon User vs Non-User) with avg spend per customer
    - Weekly revenue trend by group
    - Top categories purchased by coupon users
    - Per-customer spend distribution (mean, median, std dev)
    """
    if coupon_df is None or "HOUSEHOLD_KEY" not in coupon_df.columns:
        return {}

    coupon_users = set(coupon_df["HOUSEHOLD_KEY"].dropna().unique())
    df = df.copy()
    df["Used_Coupon"] = df["HOUSEHOLD_KEY"].isin(coupon_users).map(
        {True: "Coupon User", False: "Non-User"}
    )

    # ── Overall comparison ──
    comparison = (
        df.groupby("Used_Coupon")
        .agg(
            Total_Revenue=("SALES_VALUE", "sum"),
            Avg_Spend_Per_Txn=("SALES_VALUE", "mean"),
            Total_Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
        )
        .reset_index()
        .rename(columns={"Used_Coupon": "Group"})
    )
    comparison["Avg_Revenue_Per_Customer"] = (
        comparison["Total_Revenue"] / comparison["Unique_Customers"]
    ).round(2)
    comparison["Revenue_Share_%"] = (
        comparison["Total_Revenue"] / comparison["Total_Revenue"].sum() * 100
    ).round(1)

    # ── Weekly trend ──
    weekly_impact = (
        df.groupby(["WEEK_NO", "Used_Coupon"])["SALES_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"WEEK_NO": "Week", "Used_Coupon": "Group", "SALES_VALUE": "Revenue"})
    )

    # ── Categories for coupon users (sorted by Revenue desc, rank added) ──
    coupon_categories = (
        df[df["Used_Coupon"] == "Coupon User"]
        .groupby("COMMODITY_DESC")
        .agg(
            Revenue=("SALES_VALUE", "sum"),
            Transactions=("SALES_VALUE", "count"),
            Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
        )
        .sort_values("Revenue", ascending=False)
        .head(15)
        .reset_index()
        .rename(columns={"COMMODITY_DESC": "Category"})
    )
    coupon_categories["Revenue_Share_%"] = (
        coupon_categories["Revenue"] / coupon_categories["Revenue"].sum() * 100
    ).round(1)
    coupon_categories["Rank"] = range(1, len(coupon_categories) + 1)

    # ── Per-customer avg spend comparison ──
    per_customer = (
        df.groupby(["HOUSEHOLD_KEY", "Used_Coupon"])["SALES_VALUE"]
        .sum()
        .reset_index()
        .rename(columns={"Used_Coupon": "Group", "SALES_VALUE": "Total_Spend"})
    )
    avg_per_customer = (
        per_customer.groupby("Group")["Total_Spend"]
        .agg(["mean", "median", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean_Spend", "median": "Median_Spend", "std": "Std_Dev"})
    )
    avg_per_customer = avg_per_customer.round(2)

    return {
        "comparison": comparison,
        "weekly_impact": weekly_impact,
        "coupon_categories": coupon_categories,
        "avg_per_customer": avg_per_customer,
    }

# ─────────────────────────────────────────────
# ENGAGEMENT SCORE ANALYSIS
# ─────────────────────────────────────────────

def get_customer_engagement(df: pd.DataFrame, demo_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Calculates a 0-100 Engagement Score based on shopping Frequency and Category Diversity.
    Optionally merges demographics to profile high-engagement customers.
    """
    if "WEEK_NO" not in df.columns or "COMMODITY_DESC" not in df.columns:
        return pd.DataFrame()
    
    metrics = df.groupby("HOUSEHOLD_KEY").agg(
        Total_Spend=("SALES_VALUE", "sum"),
        Frequency=("WEEK_NO", "nunique"),
        Category_Diversity=("COMMODITY_DESC", "nunique")
    ).reset_index()
    
    f_min, f_max = metrics["Frequency"].min(), metrics["Frequency"].max()
    c_min, c_max = metrics["Category_Diversity"].min(), metrics["Category_Diversity"].max()
    
    if f_max > f_min and c_max > c_min:
        metrics["Engagement_Score"] = (((metrics["Frequency"] - f_min) / (f_max - f_min)) * 0.5 + 
                                       ((metrics["Category_Diversity"] - c_min) / (c_max - c_min)) * 0.5) * 100
    else:
        metrics["Engagement_Score"] = 0
        
    if demo_df is not None and "HOUSEHOLD_KEY" in demo_df.columns:
        metrics = metrics.merge(demo_df, on="HOUSEHOLD_KEY", how="left")
        
    metrics["Engagement_Score"] = metrics["Engagement_Score"].round(1)
    return metrics.sort_values("Engagement_Score", ascending=False)


# ─────────────────────────────────────────────
# CATEGORY GROWTH / DECLINE
# ─────────────────────────────────────────────

def get_category_growth_decline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare category revenue between the first half and second half of the dataset weeks.
    """
    if "COMMODITY_DESC" not in df.columns or "WEEK_NO" not in df.columns or "SALES_VALUE" not in df.columns:
        return pd.DataFrame()
        
    weeks = sorted(df["WEEK_NO"].dropna().unique())
    if not weeks:
        return pd.DataFrame()
        
    mid = weeks[len(weeks) // 2]
    
    recent = df[df["WEEK_NO"] > mid].groupby("COMMODITY_DESC")["SALES_VALUE"].sum().rename("Recent_Revenue")
    past = df[df["WEEK_NO"] <= mid].groupby("COMMODITY_DESC")["SALES_VALUE"].sum().rename("Past_Revenue")
    
    growth_df = pd.concat([recent, past], axis=1).fillna(0)
    growth_df["Revenue_Growth"] = growth_df["Recent_Revenue"] - growth_df["Past_Revenue"]
    
    result = growth_df.sort_values("Revenue_Growth", ascending=False).reset_index()
    result.rename(columns={"COMMODITY_DESC": "Category"}, inplace=True)
    return result.head(10)


# ─────────────────────────────────────────────
# DIRECT MARKETING EFFECT
# ─────────────────────────────────────────────

def get_direct_marketing_effect(df: pd.DataFrame, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compare engagement and revenue for Direct Marketing (TypeA) vs Mass/No Marketing.
    """
    campaign_table = next((v for k, v in dataframes.items() if "campaign_table" in k.lower()), None)
    campaign_desc = next((v for k, v in dataframes.items() if "campaign_desc" in k.lower()), None)
    
    direct_hhs = set()
    if campaign_table is not None and campaign_desc is not None:
        if "CAMPAIGN" in campaign_table.columns and "CAMPAIGN" in campaign_desc.columns:
            merged = campaign_table.merge(campaign_desc, on="CAMPAIGN", how="inner")
            desc_col = next((col for col in merged.columns if "DESCRIPTION" in col), None)
            if desc_col:
                direct_hhs = set(merged[merged[desc_col] == "TypeA"]["HOUSEHOLD_KEY"].dropna().unique())
    
    df_copy = df.copy()
    df_copy["Group"] = df_copy["HOUSEHOLD_KEY"].apply(
        lambda x: "Direct Marketing" if x in direct_hhs else "Mass/No Marketing"
    )
    
    result = df_copy.groupby("Group").agg(
        Unique_Customers=("HOUSEHOLD_KEY", "nunique"),
        Total_Revenue=("SALES_VALUE", "sum"),
        Total_Transactions=("SALES_VALUE", "count")
    ).reset_index()
    
    result["Avg_Revenue_Per_Customer"] = (result["Total_Revenue"] / result["Unique_Customers"]).round(2)
    result["Avg_Transactions_Per_Customer"] = (result["Total_Transactions"] / result["Unique_Customers"]).round(2)
    
    return result.sort_values("Total_Revenue", ascending=False)

def render_question_visualization(question_type, result_df):

    if result_df is None or result_df.empty:
        return

    try:

        # ----------------------------------------
        # Behavioral Trends
        # ----------------------------------------

        if question_type == "behavioral":

            if "Trend" in result_df.columns:

                fig = px.bar(
                    result_df,
                    x="Trend",
                    y="Total_Revenue",
                    color="Trend",
                    title="Revenue by Customer Spending Trend"
                )

                st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------
        # Campaign
        # ----------------------------------------

        elif question_type in ["campaign", "direct_marketing"]:

            fig = px.bar(
                result_df,
                x="Group",
                y="Avg_Revenue_Per_Customer",
                color="Group",
                title="Marketing Impact on Customer Revenue"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------
        # Demographic
        # ----------------------------------------

        elif question_type == "demographic":

            fig = px.bar(
                result_df.head(10),
                x="Group",
                y="Total_Revenue",
                color="Group",
                title="Revenue by Demographic Segment"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------
        # Engagement
        # ----------------------------------------

        elif question_type == "engagement":

            fig = px.scatter(
                result_df,
                x="Frequency",
                y="Engagement_Score",
                size="Total_Spend",
                hover_data=["HOUSEHOLD_KEY"],
                title="Customer Engagement Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------
        # Category
        # ----------------------------------------

        elif question_type == "category_growth":

            fig = px.bar(
                result_df.head(10),
                x=result_df.columns[0],
                y=result_df.columns[1],
                title="Category Revenue Analysis"
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")

        # ============================================
# CUSTOMER SPENDING TRENDS
# ============================================

def get_customer_trends(df):

    customer_spend = (
        df.groupby("HOUSEHOLD_KEY")["SALES_VALUE"]
        .sum()
        .reset_index()
    )

    avg_spend = customer_spend["SALES_VALUE"].mean()

    customer_spend["Trend"] = customer_spend["SALES_VALUE"].apply(
        lambda x:
            "Increasing" if x > avg_spend * 1.1
            else (
                "Decreasing" if x < avg_spend * 0.9
                else "Stable"
            )
    )

    result = (
        customer_spend.groupby("Trend")
        .agg(
            Customers=("HOUSEHOLD_KEY", "count"),
            Avg_Spend=("SALES_VALUE", "mean")
        )
        .reset_index()
    )

    return result