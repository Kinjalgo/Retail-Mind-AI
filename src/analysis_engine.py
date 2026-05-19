"""
analysis_engine.py — Pure analytics functions (pandas-based, no LLM)
"""

import pandas as pd
import numpy as np
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
# CUSTOMER BEHAVIORAL TRENDS
# ─────────────────────────────────────────────

def get_customer_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each customer's spend in first half vs second half of time period.
    Returns one row per customer with Trend label and change %.
    """
    if "WEEK_NO" not in df.columns:
        return pd.DataFrame()

    customer_week = (
        df.groupby(["HOUSEHOLD_KEY", "WEEK_NO"])["SALES_VALUE"]
        .sum()
        .reset_index()
        .sort_values(["HOUSEHOLD_KEY", "WEEK_NO"])
    )

    weeks = sorted(df["WEEK_NO"].dropna().unique())
    mid = weeks[len(weeks) // 2]

    first_half = (
        customer_week[customer_week["WEEK_NO"] <= mid]
        .groupby("HOUSEHOLD_KEY")["SALES_VALUE"].sum()
    )
    second_half = (
        customer_week[customer_week["WEEK_NO"] > mid]
        .groupby("HOUSEHOLD_KEY")["SALES_VALUE"].sum()
    )

    trend_df = pd.DataFrame({"First_Half": first_half, "Second_Half": second_half}).fillna(0)
    trend_df["Change_Pct"] = (
        (trend_df["Second_Half"] - trend_df["First_Half"])
        / trend_df["First_Half"].replace(0, np.nan)
    ) * 100
    trend_df["Trend"] = trend_df["Change_Pct"].apply(
        lambda x: "Increasing" if x > 5 else ("Decreasing" if x < -5 else "Stable")
    )
    return trend_df.reset_index()


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


def get_available_demo_dimensions(demo_df: Optional[pd.DataFrame]) -> List[str]:
    if demo_df is None:
        return []
    return [d for d in DEMO_DIMENSIONS if d in demo_df.columns]


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
