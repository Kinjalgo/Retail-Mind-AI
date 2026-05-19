"""
sql_engine.py — Natural language → SQL → execute on SQLite in-memory DB
"""

import re
import sqlite3
import pandas as pd
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────
# SCHEMA HELPER
# ─────────────────────────────────────────────

def get_schema_string(dataframes: Dict[str, pd.DataFrame]) -> str:
    """Build a human-readable schema summary for the LLM prompt."""
    lines = []
    for table_name, df in dataframes.items():
        clean_name = _table_name(table_name)
        cols = ", ".join([f"{c} ({str(df[c].dtype)})" for c in df.columns])
        lines.append(f"Table `{clean_name}`: {cols}")
    return "\n".join(lines)


def _table_name(filename: str) -> str:
    """Convert filename to SQLite-safe table name."""
    name = filename.replace(".csv", "").replace("-", "_").replace(" ", "_").lower()
    return re.sub(r"[^a-z0-9_]", "", name)


# ─────────────────────────────────────────────
# SQL CLEANER
# ─────────────────────────────────────────────

def clean_sql(raw_sql: str) -> str:
    """
    Strip markdown fences, comments, and replace unsupported syntax.
    Ensures SQLite compatibility.
    """
    sql = raw_sql.strip()

    # Remove markdown code fences
    sql = re.sub(r"```sql", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"```", "", sql)

    # Remove single-line comments
    sql = re.sub(r"--[^\n]*", "", sql)

    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)

    # Replace DATE_TRUNC (PostgreSQL) with SQLite equivalent — best effort
    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*'week'\s*,\s*([^)]+)\)",
        r"strftime('%Y-%W', \1)",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*'month'\s*,\s*([^)]+)\)",
        r"strftime('%Y-%m', \1)",
        sql,
        flags=re.IGNORECASE,
    )

    # Replace ILIKE with LIKE (SQLite doesn't have ILIKE)
    sql = re.sub(r"\bILIKE\b", "LIKE", sql, flags=re.IGNORECASE)

    # Strip leading/trailing whitespace and normalize spacing
    sql = " ".join(sql.split())

    return sql.strip()


# ─────────────────────────────────────────────
# IN-MEMORY SQLITE LOADER
# ─────────────────────────────────────────────

def load_into_sqlite(dataframes: Dict[str, pd.DataFrame]) -> sqlite3.Connection:
    """Load all DataFrames into an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for filename, df in dataframes.items():
        table = _table_name(filename)
        df.to_sql(table, conn, if_exists="replace", index=False)
    return conn


# ─────────────────────────────────────────────
# SQL EXECUTOR
# ─────────────────────────────────────────────

def execute_sql(conn: sqlite3.Connection, sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute SQL and return (result_df, error_message).
    On success: (DataFrame, None)
    On failure: (None, error_string)
    """
    try:
        result = pd.read_sql_query(sql, conn)
        return result, None
    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
# SQL GENERATION PROMPT BUILDER
# ─────────────────────────────────────────────

def build_sql_prompt(question: str, dataframes: Dict[str, pd.DataFrame]) -> str:
    """Build a strict prompt for the LLM to generate SQLite-compatible SQL."""
    schema = get_schema_string(dataframes)
    table_names = [_table_name(k) for k in dataframes.keys()]

    return f"""You are a SQLite SQL expert. Generate a single, valid SQLite query.

DATABASE SCHEMA:
{schema}

AVAILABLE TABLES: {', '.join(table_names)}

STRICT RULES:
1. Output ONLY the raw SQL query — no markdown, no explanation, no comments
2. Use only SQLite-compatible syntax
3. Do NOT use DATE_TRUNC, ILIKE, or PostgreSQL-specific functions
4. Use strftime() for date operations
5. Always use table aliases when joining
6. Limit results to 100 rows unless the question asks for all
7. Column names are CASE-SENSITIVE — use exact names from schema
8. Use SUM(SALES_VALUE) from the transaction table to calculate revenue/sales
9. Link tables together using HOUSEHOLD_KEY or PRODUCT_ID when joining
10. For marketing, campaign, or causal questions, join the relevant campaign or causal tables

QUESTION: {question}

SQL:"""


# ─────────────────────────────────────────────
# TABLE NAME UTILITY (PUBLIC)
# ─────────────────────────────────────────────

def get_table_name(filename: str) -> str:
    return _table_name(filename)
