"""
data_loader.py — CSV ingestion, merging, and schema normalization
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Optional


COLUMN_ALIASES = {
    "SALES_VALUE": ["SALES_VALUE", "SALE_VALUE", "REVENUE", "AMOUNT", "TOTAL"],
    "HOUSEHOLD_KEY": ["HOUSEHOLD_KEY", "HH_KEY", "CUSTOMER_ID", "CUST_ID", "HH_ID"],
    "PRODUCT_ID": ["PRODUCT_ID", "PROD_ID", "ITEM_ID", "SKU"],
    "COMMODITY_DESC": ["COMMODITY_DESC", "CATEGORY", "PRODUCT_CATEGORY", "DEPT", "DEPARTMENT"],
    "WEEK_NO": ["WEEK_NO", "WEEK", "WEEK_NUM", "PERIOD"],
    "AGE_DESC": ["AGE_DESC", "CLASSIFICATION_1", "AGE_GROUP", "AGE", "AGE_RANGE"],
    "MARITAL_STATUS_CODE": ["MARITAL_STATUS_CODE", "CLASSIFICATION_2", "MARITAL", "MARITAL_STATUS"],
    "INCOME_DESC": ["INCOME_DESC", "CLASSIFICATION_3", "INCOME", "INCOME_RANGE"],
    "HOUSEHOLD_SIZE_DESC": ["HOUSEHOLD_SIZE_DESC", "CLASSIFICATION_4", "HOUSEHOLD_SIZE", "HH_SIZE"],
    "HH_COMP_DESC": ["HH_COMP_DESC", "CLASSIFICATION_5", "HOUSEHOLD_COMP", "HH_COMP"],
    "KID_CATEGORY_DESC": ["KID_CATEGORY_DESC", "CLASSIFICATION_7", "KIDS", "CHILDREN"],
    "CAMPAIGN": ["CAMPAIGN", "CAMPAIGN_ID"],
    "COUPON_UPC": ["COUPON_UPC", "COUPON_ID", "UPC"]
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase + strip all column names, then apply aliases."""
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


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality strings to categories to prevent OOM."""
    id_cols = ["HOUSEHOLD_KEY", "PRODUCT_ID", "BASKET_ID", "STORE_ID", "CAMPAIGN", "COUPON_UPC"]
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'object' and col not in id_cols:
            if df[col].nunique() / max(len(df), 1) < 0.3:
                df[col] = df[col].astype('category')
    import gc
    gc.collect()
    return df


def load_dataframes(uploaded_files) -> Dict[str, pd.DataFrame]:
    """Load all uploaded CSVs into a dict keyed by filename."""
    dataframes = {}
    for file in uploaded_files:
        try:
            df = pd.read_csv(file)
            df = normalize_columns(df)
            
            # Ensure ID columns are consistently strings across ALL dataframes before merging
            id_cols = ["HOUSEHOLD_KEY", "PRODUCT_ID", "BASKET_ID", "STORE_ID", "CAMPAIGN", "COUPON_UPC"]
            for col in id_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(str).str.replace(r'\.0$', '', regex=True)
                    
            df = optimize_memory(df)
            dataframes[file.name] = df
        except Exception as e:
            st.warning(f"Could not load {file.name}: {e}")
    return dataframes


def build_main_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Merge transaction + product + demographic data.
    Gracefully handles missing files.
    Includes business logic for calculating prices and pseudo-dates.
    """
    trans_key = next((k for k in dataframes if "transaction" in k.lower()), None)
    prod_key = next((k for k in dataframes if "product" in k.lower()), None)
    demo_key = next((k for k in dataframes if "demographic" in k.lower() or "demo" in k.lower()), None)

    if trans_key:
        df = dataframes[trans_key].copy()
        
        # 3. Primary & Foreign Keys (PRE-MERGE STRATEGY)
        if prod_key and "PRODUCT_ID" in dataframes[prod_key].columns:
            prod_df = dataframes[prod_key].copy()
            df = df.merge(prod_df, on="PRODUCT_ID", how="left")
            
        if demo_key and "HOUSEHOLD_KEY" in dataframes[demo_key].columns:
            demo_df = dataframes[demo_key].copy()
            df = df.merge(demo_df, on="HOUSEHOLD_KEY", how="left")
    elif dataframes:
        df = list(dataframes.values())[0].copy()
    else:
        return None

    # Ensure required columns exist
    if "COMMODITY_DESC" not in df.columns:
        df["COMMODITY_DESC"] = "UNKNOWN"
    if "WEEK_NO" not in df.columns:
        df["WEEK_NO"] = range(len(df))
    if "HOUSEHOLD_KEY" not in df.columns:
        df["HOUSEHOLD_KEY"] = 0

    # 5. Missing Values Handling
    missing_values = ["", "NA", "NULL", "null", "NaN", None]
    df.replace(missing_values, np.nan, inplace=True)

    # Fill financial & numeric columns with 0 where NA
    for col in ["COUPON_DISC", "COUPON_MATCH_DISC", "RETAIL_DISC", "SALES_VALUE", "QUANTITY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 4. Data Types (Time Handling)
    if "DAY" in df.columns:
        df["DAY"] = pd.to_numeric(df["DAY"], errors="coerce")
        df["PSEUDO_DATE"] = pd.to_datetime("2020-01-01") + pd.to_timedelta(df["DAY"], unit="D")
        
    if "WEEK_NO" in df.columns:
        df["WEEK_NO"] = pd.to_numeric(df["WEEK_NO"], errors="coerce")

    # 6. Business Logic (Actual Price Calculation)
    if all(c in df.columns for c in ["SALES_VALUE", "QUANTITY", "COUPON_MATCH_DISC"]):
        # Prevent division by zero
        df["ACTUAL_PRICE"] = np.where(df["QUANTITY"] == 0, 0, (df["SALES_VALUE"] - df["COUPON_MATCH_DISC"]) / df["QUANTITY"])
        
        if "RETAIL_DISC" in df.columns:
            df["LOYALTY_PRICE"] = np.where(
                df["QUANTITY"] == 0, 
                0, 
                (df["SALES_VALUE"] - (df["RETAIL_DISC"] + df["COUPON_MATCH_DISC"])) / df["QUANTITY"]
            )

    df = optimize_memory(df)
    return df


def get_demo_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return the demographic dataframe if available."""
    demo_key = next((k for k in dataframes if "demographic" in k.lower() or "demo" in k.lower()), None)
    if demo_key:
        return dataframes[demo_key]
    return None


def get_coupon_dataframe(dataframes: Dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Return the coupon/campaign dataframe if available."""
    
    # Prioritize tables that actually link to households (redemptions or campaign_table)
    coupon_key = next((k for k in dataframes if "redempt" in k.lower()), None)
    if not coupon_key:
        coupon_key = next((k for k in dataframes if "campaign_table" in k.lower()), None)
    if not coupon_key:
        coupon_key = next((k for k in dataframes if any(w in k.lower() for w in ["coupon", "campaign", "causal", "marketing"])), None)
    
    if coupon_key:
        df = dataframes[coupon_key].copy()
        
        # Optional advanced joins for coupon enrichment
        prod_key = next((k for k in dataframes if "product" in k.lower()), None)
        if prod_key and "PRODUCT_ID" in df.columns and "PRODUCT_ID" in dataframes[prod_key].columns:
            df = df.merge(dataframes[prod_key], on="PRODUCT_ID", how="left")
            
        desc_key = next((k for k in dataframes if "desc" in k.lower() and "campaign" in k.lower()), None)
        if desc_key and "CAMPAIGN" in df.columns and "CAMPAIGN" in dataframes[desc_key].columns:
            df = df.merge(dataframes[desc_key], on="CAMPAIGN", how="left")
            
        return df
    return None
