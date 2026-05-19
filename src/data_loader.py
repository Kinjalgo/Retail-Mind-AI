"""
data_loader.py — CSV ingestion, merging, and schema normalisation.

Two loading modes:
  1. load_from_directory(data_dir)  — reads directly from disk (no upload size limit)
  2. load_dataframes(uploaded_files) — Streamlit file-uploader (kept for custom datasets)

causal_data.csv is excluded from the default load because it is 664 MB.
Call load_causal_data(data_dir) separately only when needed.
"""

import os
import pandas as pd
import streamlit as st
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ALIASES
# Maps raw CSV column names (including hh_demographic's obfuscated names)
# to the canonical names used everywhere in the codebase.
# ─────────────────────────────────────────────────────────────────────────────

COLUMN_ALIASES = {
    "SALES_VALUE":          ["SALES_VALUE", "SALE_VALUE", "REVENUE", "AMOUNT", "TOTAL"],
    "HOUSEHOLD_KEY":        ["HOUSEHOLD_KEY", "HH_KEY", "CUSTOMER_ID", "CUST_ID", "HH_ID"],
    "PRODUCT_ID":           ["PRODUCT_ID", "PROD_ID", "ITEM_ID", "SKU"],
    "COMMODITY_DESC":       ["COMMODITY_DESC", "CATEGORY", "PRODUCT_CATEGORY", "DEPT", "DEPARTMENT"],
    "WEEK_NO":              ["WEEK_NO", "WEEK", "WEEK_NUM", "PERIOD"],
    # hh_demographic uses generic "classification_N" column names in the CSV.
    # Mapped to meaningful canonical names per Dunnhumby documentation.
    "AGE_DESC":             ["AGE_DESC", "CLASSIFICATION_1", "AGE_GROUP", "AGE", "AGE_RANGE"],
    "MARITAL_STATUS_CODE":  ["MARITAL_STATUS_CODE", "CLASSIFICATION_2", "MARITAL", "MARITAL_STATUS"],
    "INCOME_DESC":          ["INCOME_DESC", "CLASSIFICATION_3", "INCOME", "INCOME_RANGE"],
    "HOUSEHOLD_SIZE_DESC":  ["HOUSEHOLD_SIZE_DESC", "CLASSIFICATION_4", "HH_SIZE", "HOUSEHOLD_SIZE"],
    "HH_COMP_DESC":         ["HH_COMP_DESC", "CLASSIFICATION_5", "HH_COMP", "HOUSEHOLD_COMP"],
}

# Files to skip in the automatic directory scan (loaded separately or not at all)
_SKIP_FILES = {"causal_data.csv"}

# Dtype overrides per filename — reduce memory for large files
_DTYPE_OVERRIDES: Dict[str, dict] = {
    "transaction_data.csv": {
        "household_key":      "int32",
        "basket_id":          "int64",
        "day":                "int16",
        "product_id":         "int32",
        "quantity":           "int16",
        "sales_value":        "float32",
        "store_id":           "int16",
        "retail_disc":        "float32",
        "coupon_disc":        "float32",
        "coupon_match_disc":  "float32",
        "trans_time":         "int32",
        "week_no":            "int8",
    },
    "product.csv": {
        "PRODUCT_ID":         "int32",
        "MANUFACTURER":       "int32",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    existing = set(df.columns)
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical not in existing:
            for alias in aliases:
                if alias.upper() in existing:
                    df.rename(columns={alias.upper(): canonical}, inplace=True)
                    break
    return df


def _read_csv(path: str, filename: str) -> pd.DataFrame:
    dtypes = _DTYPE_OVERRIDES.get(filename, {})
    # Lower-case dtype keys to match raw CSV headers before normalisation
    dtypes_lower = {k.lower(): v for k, v in dtypes.items()}
    try:
        df = pd.read_csv(path, dtype=dtypes_lower, low_memory=False)
    except Exception:
        df = pd.read_csv(path, low_memory=False)
    return normalize_columns(df)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — LOAD FROM DIRECTORY  (primary mode, no size limit)
# ─────────────────────────────────────────────────────────────────────────────

def load_from_directory(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Read every CSV in data_dir except causal_data.csv (loaded separately).
    Returns a dict keyed by filename.
    """
    dataframes: Dict[str, pd.DataFrame] = {}
    if not os.path.isdir(data_dir):
        st.error(f"Data folder not found: {data_dir}")
        return dataframes

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv") or fname in _SKIP_FILES:
            continue
        path = os.path.join(data_dir, fname)
        try:
            df = _read_csv(path, fname)
            dataframes[fname] = df
        except Exception as e:
            st.warning(f"Could not load {fname}: {e}")

    return dataframes


def load_causal_data(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Load causal_data.csv with memory-efficient dtypes.
    Only call this when the user explicitly requests display/mailer analysis.
    """
    path = os.path.join(data_dir, "causal_data.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(
            path,
            dtype={
                "product_id": "int32",
                "store_id":   "int16",
                "week_no":    "int8",
                "display":    "str",
                "mailer":     "str",
            },
            low_memory=False,
        )
        return normalize_columns(df)
    except Exception as e:
        st.warning(f"Could not load causal_data.csv: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — STREAMLIT FILE UPLOADER  (fallback for custom datasets)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataframes(uploaded_files) -> Dict[str, pd.DataFrame]:
    dataframes: Dict[str, pd.DataFrame] = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file, low_memory=False)
            df = normalize_columns(df)
            dataframes[file.name] = df
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")
    return dataframes


# ─────────────────────────────────────────────────────────────────────────────
# MERGE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_main_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Merge transaction_data with product to add COMMODITY_DESC, DEPARTMENT, BRAND.
    Falls back gracefully if either file is missing.
    """
    trans_key = next((k for k in dataframes if "transaction" in k.lower()), None)
    prod_key  = next((k for k in dataframes if "product" in k.lower()), None)

    if trans_key:
        df = dataframes[trans_key].copy()
        if prod_key and "PRODUCT_ID" in dataframes[prod_key].columns:
            prod_cols = ["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC",
                         "SUB_COMMODITY_DESC", "BRAND"]
            prod_cols = [c for c in prod_cols if c in dataframes[prod_key].columns]
            df = df.merge(dataframes[prod_key][prod_cols], on="PRODUCT_ID", how="left")
    elif dataframes:
        df = list(dataframes.values())[0].copy()
    else:
        return None

    if "COMMODITY_DESC" not in df.columns:
        df["COMMODITY_DESC"] = "UNKNOWN"
    if "WEEK_NO" not in df.columns:
        df["WEEK_NO"] = range(len(df))
    if "HOUSEHOLD_KEY" not in df.columns:
        df["HOUSEHOLD_KEY"] = 0

    for col in ["SALES_VALUE", "WEEK_NO", "HOUSEHOLD_KEY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_demo_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    key = next((k for k in dataframes if "demographic" in k.lower() or "demo" in k.lower()), None)
    return dataframes[key] if key else None


def get_coupon_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    key = next((k for k in dataframes if "redempt" in k.lower()), None)
    if key:
        return dataframes[key]
    key = next((k for k in dataframes if "coupon" in k.lower()), None)
    return dataframes[key] if key else None
