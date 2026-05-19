"""
ai_agents.py — Claude API-powered SQL generator and insight engine
Replaces CrewAI + Ollama with direct Anthropic SDK calls + prompt caching.
"""

import anthropic
import pandas as pd
from typing import Dict, Optional

from sql_engine import build_sql_prompt, clean_sql


# ─────────────────────────────────────────────
# CLIENT FACTORY
# ─────────────────────────────────────────────

def get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


# ─────────────────────────────────────────────
# DATA DICTIONARY  (corrected from Dunnhumby PDF + actual CSV columns)
# Cached via prompt caching — only charged once per session.
# ─────────────────────────────────────────────

DATA_DICTIONARY = """
DATASET: Dunnhumby "The Complete Journey" — 2,500 households, 2 years of grocery transactions.

TABLE: transaction_data  (2.5M rows — the core fact table)
  household_key     — unique household ID (FK → hh_demographic, campaign_table)
  basket_id         — unique purchase occasion / receipt
  day               — day number (1–711)
  product_id        — unique product ID (FK → product, coupon, causal_data)
  quantity          — units purchased
  sales_value       — dollars retailer receives (NOT the price the customer paid)
  store_id          — unique store ID
  retail_disc       — loyalty card discount applied (negative, e.g. -0.60)
  coupon_disc       — manufacturer coupon discount (negative)
  coupon_match_disc — retailer match of manufacturer coupon (negative)
  trans_time        — time of day (HHMM integer, e.g. 1631 = 4:31 PM)
  week_no           — week number (1–102)

  Price formulas:
    Loyalty card price per unit     = (sales_value - (retail_disc + coupon_match_disc)) / quantity
    Non-loyalty card price per unit = (sales_value - coupon_match_disc) / quantity
    Actual price customer paid      = sales_value + coupon_disc  (coupon_disc is negative)

TABLE: hh_demographic  (~800 of 2,500 households have demographic data)
  HOUSEHOLD_KEY       — unique household ID
  AGE_DESC            — age group: "Age Group1" (youngest) through "Age Group6" (oldest)
  MARITAL_STATUS_CODE — marital status: "A" (Married), "B" (Single/Unknown), "U" (Unknown)
                        NOTE: raw CSV shows values X, Y, Z — treat as ordered categories
  INCOME_DESC         — income level: "Level1" (lowest) through "Level12" (highest income)
  HOMEOWNER_DESC      — "Homeowner", "Renter", "Probable Owner", "Probable Renter", "Unknown/Other"
  HH_COMP_DESC        — household composition: "Group1" through "Group5" (ordered)
  HOUSEHOLD_SIZE_DESC — household size: "1", "2", "3", "4", "5+"
  KID_CATEGORY_DESC   — number of children: "None/Unknown", "1", "2", "3+"

TABLE: product  (92K products)
  PRODUCT_ID           — unique product ID
  DEPARTMENT           — broad dept (e.g. GROCERY, MEAT, PRODUCE, DAIRY)
  COMMODITY_DESC       — product category (e.g. "FLUID MILK PRODUCTS", "BEEF")
  SUB_COMMODITY_DESC   — subcategory (e.g. "CHEESE - NATURAL")
  MANUFACTURER         — manufacturer code (integer)
  BRAND                — "National" or "Private"
  CURR_SIZE_OF_PRODUCT — package size (string, not available for all products)

TABLE: campaign_table  (which households received which campaigns)
  HOUSEHOLD_KEY — unique household ID
  CAMPAIGN      — campaign ID (1–30)
  DESCRIPTION   — campaign type: "TypeA", "TypeB", or "TypeC"

TABLE: campaign_desc  (campaign date ranges)
  CAMPAIGN    — campaign ID (1–30)
  DESCRIPTION — campaign type: TypeA, TypeB, TypeC
  START_DAY   — first day of campaign (matches transaction_data.day)
  END_DAY     — last day of campaign

TABLE: coupon  (which products each coupon covers)
  CAMPAIGN   — campaign ID
  COUPON_UPC — unique coupon identifier
  PRODUCT_ID — product the coupon can be redeemed for (one coupon may cover many products)

TABLE: coupon_redempt  (actual coupon redemptions per household)
  HOUSEHOLD_KEY — unique household ID
  DAY           — day coupon was redeemed
  COUPON_UPC    — coupon that was redeemed
  CAMPAIGN      — campaign the coupon belongs to

TABLE: causal_data  (in-store display and mailer promotions by product/store/week)
  PRODUCT_ID — unique product ID
  STORE_ID   — unique store ID
  WEEK_NO    — week number
  DISPLAY    — display location (0=not displayed; 1=Store Front; 2=Store Rear;
               3=Front End Cap; 4=Mid-Aisle End Cap; 5=Rear End Cap;
               6=Side-Aisle End Cap; 7=In-Aisle; 9=Secondary Location; A=In-Shelf)
  MAILER     — weekly mailer placement (0=not in ad; A=Interior page feature;
               C=Interior page line item; D=Front page feature; F=Back page feature;
               H=Wrap front; J=Wrap interior coupon; L=Wrap back; P=Interior coupon;
               X=Free on interior page; Z=Free on front/back page)

KEY JOIN PATHS:
  transactions ↔ products:        transaction_data.product_id   = product.PRODUCT_ID
  transactions ↔ demographics:    transaction_data.household_key = hh_demographic.HOUSEHOLD_KEY
  transactions ↔ campaigns sent:  transaction_data.household_key = campaign_table.HOUSEHOLD_KEY
  transactions ↔ causal/promo:    transaction_data.product_id   = causal_data.PRODUCT_ID
                                   AND transaction_data.store_id  = causal_data.STORE_ID
                                   AND transaction_data.week_no   = causal_data.WEEK_NO
  coupon redemptions ↔ campaigns: coupon_redempt.CAMPAIGN = campaign_table.CAMPAIGN
                                   AND coupon_redempt.HOUSEHOLD_KEY = campaign_table.HOUSEHOLD_KEY
"""


# ─────────────────────────────────────────────
# QUESTION CLASSIFIER  (keyword-based, no LLM needed)
# ─────────────────────────────────────────────

KEYWORD_MAP = {
    "behavioral": ["increas", "decreas", "growing", "declining", "spending more", "spending less", "trend"],
    "campaign":   ["coupon", "campaign", "promotion", "discount", "marketing", "offer", "redempt", "mailer"],
    "demographic":["age", "income", "demographic", "household", "family", "segment", "marital", "homeowner", "kids"],
    "category":   ["category", "product", "item", "commodity", "department", "buying", "brand"],
}

def classify_question(question: str) -> str:
    q = question.lower()
    for k, v in KEYWORD_MAP.items():
        if any(word in q for word in v):
            return k
    return "trend"


# ─────────────────────────────────────────────
# SQL GENERATOR
# ─────────────────────────────────────────────

def generate_sql(question: str, dataframes: Dict[str, pd.DataFrame], client: anthropic.Anthropic) -> str:
    schema_prompt = build_sql_prompt(question, dataframes)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": DATA_DICTIONARY,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    "You are a SQLite SQL expert. "
                    "Output ONLY the raw SQL query — no markdown fences, no explanation, no comments. "
                    "Use only SQLite-compatible syntax. Never use DATE_TRUNC or ILIKE."
                ),
            },
        ],
        messages=[{"role": "user", "content": schema_prompt}],
    )
    return clean_sql(response.content[0].text)


# ─────────────────────────────────────────────
# INSIGHT GENERATOR
# ─────────────────────────────────────────────

_TYPE_INSTRUCTIONS = {
    "behavioral": (
        "You are analyzing customers whose spending is increasing or decreasing over time.\n"
        "You MUST answer: (1) how many customers in each trend group, "
        "(2) top customers by spend, (3) which categories they buy most — Row 1 is always the top category."
    ),
    "campaign": (
        "You are comparing Coupon Users vs Non-Users.\n"
        "Start with YES or NO: do coupon users spend more per customer?\n"
        "Then give exact numbers for both groups."
    ),
    "demographic": (
        "You are analyzing which demographic segments drive the most revenue.\n"
        "Identify: the top revenue group, the highest avg-spend group, and any underperforming groups."
    ),
    "category": (
        "You are analyzing product categories by revenue.\n"
        "Row 1 is the highest-revenue category. Always lead with Row 1 and list the top 5 with exact values."
    ),
    "trend": (
        "You are analyzing spending over time.\n"
        "Identify the peak week, the lowest week, and whether the overall trend is growing or declining."
    ),
}


def generate_insight(
    question: str,
    result_df: pd.DataFrame,
    question_type: str,
    client: anthropic.Anthropic,
    kpi_context: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:
    numbered_rows = "\n".join(
        f"Row {i + 1}: {dict(row)}" for i, row in result_df.head(15).iterrows()
    )

    kpi_block   = f"\nKPI SUMMARY:\n{kpi_context}"      if kpi_context   else ""
    extra_block = f"\nADDITIONAL DATA:\n{extra_context}" if extra_context else ""
    type_instr  = _TYPE_INSTRUCTIONS.get(question_type, _TYPE_INSTRUCTIONS["trend"])

    prompt = f"""QUESTION: {question}
ANALYSIS TYPE: {question_type}

DATA (Row 1 = highest / most important):
{numbered_rows}
{kpi_block}
{extra_block}

STRICT RULES:
- Use ONLY the numbers in the data above — never invent values
- Row 1 is always the top result — always mention it first
- If the question is yes/no — answer that first, then explain

FORMAT YOUR RESPONSE AS:

**Key Finding:** [direct 1–2 sentence answer]

**Top Insights:**
- #1: [Row 1 with exact value]
- #2: [Row 2 with exact value]
- #3: [Row 3 with exact value]

**Business Impact:** [what this means for the retailer in 2–3 sentences]

{type_instr}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=900,
        system=[
            {
                "type": "text",
                "text": DATA_DICTIONARY,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    "You are a senior retail data analyst. "
                    "Provide accurate, grounded insights using only the data given. "
                    "Never hallucinate values or change rankings."
                ),
            },
        ],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
