"""
utils.py — Shared formatting and display helpers
"""

import pandas as pd
import streamlit as st


def fmt_currency(value: float) -> str:
    """Format a number as USD currency string."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.2f}"


def fmt_number(value: float) -> str:
    """Format a plain number with commas."""
    return f"{value:,.0f}"


def fmt_percent(value: float) -> str:
    """Format a percent."""
    return f"{value:+.1f}%"


def show_dataframe(df: pd.DataFrame, title: str = "", height: int = 300):
    """Display a styled dataframe with an optional title."""
    if title:
        st.markdown(f"**{title}**")
    st.dataframe(df, use_container_width=True, height=height)


def truncate_string(s: str, max_len: int = 30) -> str:
    return s if len(s) <= max_len else s[:max_len] + "…"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default
