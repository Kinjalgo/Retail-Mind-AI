import re
import pandas as pd
from typing import Dict, Optional, Any
from crewai import Agent, Task, Crew, Process, LLM

from sql_engine import build_sql_prompt, clean_sql


# ─────────────────────────────────────────────
# LLM FACTORY
# ─────────────────────────────────────────────

def get_llm(model: str = "ollama/qwen2.5-coder:3b", base_url: str = "http://localhost:11434") -> LLM:
    return LLM(model=model, base_url=base_url, temperature=0)


# ─────────────────────────────────────────────
# QUESTION CLASSIFIER
# ─────────────────────────────────────────────

def classify_question(question: str, llm: LLM) -> str:
    prompt = f"""
    You are an intelligent routing agent for a retail data analytics platform.
    Read the user's natural language question and decide which analytical tool should handle it.
    
    AVAILABLE TOOLS:
    1. 'behavioral' - Use this for questions about customers spending more vs less over time.
    2. 'campaign' - Use this for questions comparing coupon users vs non-users.
    3. 'demographic' - Use this for questions about age, income, or demographic factors.
    4. 'category' - Use this for questions about top-performing products.
    5. 'trend' - Use this for questions about weekly/monthly sales trends over time.
    6. 'custom' - Use this for advanced attribution, what influenced a purchase, direct marketing impact, category growth vs decline, or anything requiring custom SQL.
    
    QUESTION: "{question}"
    
    Respond with ONLY ONE word from the list above that best matches the intent.
    """
    
    agent = Agent(
        role="Semantic Router",
        goal="Understand natural language and route to the correct tool.",
        backstory="You output exactly one word from the available tools list.",
        llm=llm,
        verbose=False
    )
    task = Task(description=prompt, expected_output="A single word.", agent=agent)
    result = str(Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()).strip().lower()
    
    valid_categories = ["behavioral", "campaign", "demographic", "category", "trend", "custom"]
    for cat in valid_categories:
        if cat in result:
            return cat
            
    return "custom"


# ─────────────────────────────────────────────
# SQL GENERATOR
# ─────────────────────────────────────────────

DATA_DICTIONARY = """
DATA DICTIONARY & VARIABLE DESCRIPTIONS:
Use these descriptions to map business concepts (like 'direct marketing', 'engagement', 'revenue') to the correct tables, columns, and formulas.

- transaction_data Table: This table contains all products purchased by households. Essentially store receipts. SALES_VALUE = revenue/spend.
  * HOUSEHOLD_KEY: Uniquely identifies each household (FK to hh_demographic, campaign_table)
  * BASKET_ID: Unique shopping trip ID
  * DAY: Day when transaction occurred
  * PRODUCT_ID: Uniquely identifies each product (FK to product, coupon)
  * QUANTITY: Units purchased
  * SALES_VALUE: Dollars retailer receives from the sale (NOT the actual price paid by the customer)
  * STORE_ID: Store identifier
  * COUPON_MATCH_DISC: Retailer-matched manufacturer coupon discount
  * COUPON_DISC: Manufacturer coupon discount
  * RETAIL_DISC: Loyalty card discount
  * TRANS_TIME: Time of day
  * WEEK_NO: Week of the transaction. Ranges 1 - 102

- hh_demographic Table: This table provides a representation of demographic information for a portion of households.
  * HOUSEHOLD_KEY: PK
  * CLASSIFICATION_1: Demographic segment (Group1-Group6, ordered)
  * CLASSIFICATION_2: Demographic segment (X, Y, Z)
  * CLASSIFICATION_3: Demographic segment (Level1-Level12, ordered)
  * CLASSIFICATION_4: Demographic segment (1 through 5+, ordered)
  * CLASSIFICATION_5: Demographic segment (Group1-Group6, ordered)
  * CLASSIFICATION_6: Demographic segment (Group1-Group5, ordered)
  * CLASSIFICATION_7: Demographic segment (1, 2, 3, None/Unknown)

- campaign_table Table: This table lists the campaigns received by each household in the dataset.
  * HOUSEHOLD_KEY: FK -> transaction_data
  * CAMPAIGN: FK -> campaign_desc (1-30)
  * DESCRIPTION: TypeA, TypeB, or TypeC

- campaign_desc Table: This table gives the length of time for which a campaign runs. Any coupons received as part of a campaign are valid within the dates contained in this table.
  * CAMPAIGN: PK (1-30)
  * DESCRIPTION: TypeA, TypeB, or TypeC
  * START_DAY: Campaign start day
  * END_DAY: Campaign end day

- coupon Table: This table lists all the coupons sent to customers as part of a campaign, as well as the products for which each coupon is redeemable.
  * CAMPAIGN: FK -> campaign_desc
  * COUPON_UPC: PK - unique coupon ID
  * PRODUCT_ID: FK -> product

- coupon_redempt Table: This table identifies the coupons that each household redeemed.
  * HOUSEHOLD_KEY: FK -> transaction_data
  * DAY: Day of redemption
  * COUPON_UPC: FK -> coupon
  * CAMPAIGN: FK -> campaign_desc & coupon

- product Table: This table contains information on each product sold such as type of product, national or private label and a brand identifier.
  * PRODUCT_ID: PK
  * DEPARTMENT: Broadest grouping
  * COMMODITY_DESC: Mid-level grouping
  * SUB_COMMODITY_DESC: Most granular grouping
  * MANUFACTURER: Manufacturer code
  * BRAND: National or Private label
  * CURR_SIZE_OF_PRODUCT: Package size

- causal_data Table: This table signifies whether a given product was featured in the weekly mailer or was part of an in-store display. Use this to analyze customer behavioral responses to marketing visibility.
  * PRODUCT_ID: FK -> product (via transaction_data)
  * STORE_ID: FK -> transaction_data
  * WEEK_NO: FK -> transaction_data
  * DISPLAY: In-store display location code (0: Not on Display, 1: Store Front, 2: Store Rear, 3: Front End Cap, 4: Mid-Aisle End Cap, 5: Rear End Cap, 6: Side-Aisle End Cap, 7: In-Aisle, 9: Secondary Location Display, A: In-Shelf)
  * MAILER: Weekly mailer placement code (0: Not on ad, A: Interior page feature, C: Interior page line item, D: Front page feature, F: Back page feature, H: Wrap front feature, J: Wrap interior coupon, L: Wrap back feature, P: Interior page coupon, X: Free on interior page, Z: Free on front page/back page/wrap)

EXACT JOIN RELATIONSHIPS (CRITICAL RULES):
1. TO JOIN transactions to demographics: `JOIN hh_demographic ON transaction_data.HOUSEHOLD_KEY = hh_demographic.HOUSEHOLD_KEY`
2. TO JOIN transactions to campaigns: `JOIN campaign_table ON transaction_data.HOUSEHOLD_KEY = campaign_table.HOUSEHOLD_KEY`
3. TO JOIN transactions to products: `JOIN product ON transaction_data.PRODUCT_ID = product.PRODUCT_ID`
4. TO JOIN campaigns to campaign info: `JOIN campaign_desc ON campaign_table.CAMPAIGN = campaign_desc.CAMPAIGN`

PREVENTING FAN-OUT & INCORRECT AGGREGATIONS (CRITICAL SQL RULES):
- Coupon Users vs Non-Users: DO NOT directly JOIN transaction_data to coupon_redempt or campaign_table! This multiplies rows and inflates revenue.
- To compare revenue for coupon redeemers vs non-redeemers, you MUST use a LEFT JOIN with a DISTINCT subquery. 
  Example:
  SELECT 
      CASE WHEN cr.HOUSEHOLD_KEY IS NOT NULL THEN 'Coupon User' ELSE 'Non-User' END AS user_type,
      SUM(td.SALES_VALUE) AS total_revenue
  FROM transaction_data td
  LEFT JOIN (SELECT DISTINCT HOUSEHOLD_KEY FROM coupon_redempt) cr 
      ON td.HOUSEHOLD_KEY = cr.HOUSEHOLD_KEY
  GROUP BY user_type;

BUSINESS RULES & CALCULATIONS:
- Engagement: If a user asks about "engagement", calculate it by counting total transactions (`COUNT(BASKET_ID)`), unique customers (`COUNT(DISTINCT HOUSEHOLD_KEY)`), or total spend (`SUM(SALES_VALUE)`).
- Direct Marketing / Targeted Campaigns: Maps to `campaign_desc.DESCRIPTION = 'TypeA'` or `campaign_table.DESCRIPTION = 'TypeA'`. Filter for this when asked about direct marketing.
- Mass Marketing / Broadcast Campaigns: Maps to `DESCRIPTION IN ('TypeB', 'TypeC')`.
- Price Formulas: 
   - Loyalty card price/unit = (SALES_VALUE - RETAIL_DISC - COUPON_MATCH_DISC) / QUANTITY
   - Customer actually paid = SALES_VALUE - COUPON_DISC

EXACT SQL TEMPLATES FOR SPECIFIC QUESTIONS:
If the user asks "Which categories are driving growth vs decline?":
  SELECT p.COMMODITY_DESC as category, 
         SUM(CASE WHEN td.WEEK_NO > 50 THEN td.SALES_VALUE ELSE 0 END) as recent_revenue,
         SUM(CASE WHEN td.WEEK_NO <= 50 THEN td.SALES_VALUE ELSE 0 END) as past_revenue,
         (SUM(CASE WHEN td.WEEK_NO > 50 THEN td.SALES_VALUE ELSE 0 END) - SUM(CASE WHEN td.WEEK_NO <= 50 THEN td.SALES_VALUE ELSE 0 END)) as revenue_growth
  FROM transaction_data td
  JOIN product p ON td.PRODUCT_ID = p.PRODUCT_ID
  GROUP BY p.COMMODITY_DESC
  ORDER BY revenue_growth DESC LIMIT 10;

If the user asks "Is there evidence that direct marketing improves engagement?":
  SELECT 
      CASE WHEN c.DESCRIPTION = 'TypeA' THEN 'Direct Marketing User' ELSE 'Non-Direct Marketing' END AS user_group,
      COUNT(DISTINCT td.HOUSEHOLD_KEY) AS unique_customers,
      SUM(td.SALES_VALUE) AS total_revenue,
      SUM(td.SALES_VALUE) / COUNT(DISTINCT td.HOUSEHOLD_KEY) AS avg_spend_per_customer
  FROM transaction_data td
  LEFT JOIN (
      SELECT ct.HOUSEHOLD_KEY, cd.DESCRIPTION 
      FROM campaign_table ct 
      JOIN campaign_desc cd ON ct.CAMPAIGN = cd.CAMPAIGN 
      WHERE cd.DESCRIPTION = 'TypeA'
      GROUP BY ct.HOUSEHOLD_KEY
  ) c ON td.HOUSEHOLD_KEY = c.HOUSEHOLD_KEY
  GROUP BY user_group;

If the user asks about marketing attribution (e.g., "What influenced a purchase?", "Did the customer buy because of a coupon, campaign, display, or promotion?"):
  SELECT CASE WHEN cr.HOUSEHOLD_KEY IS NOT NULL THEN 'Coupon' WHEN cd.PRODUCT_ID IS NOT NULL THEN 'Display/Mailer' WHEN ct.HOUSEHOLD_KEY IS NOT NULL THEN 'Campaign' ELSE 'Organic (Anyway)' END AS attribution_source, SUM(td.SALES_VALUE) as total_revenue, COUNT(DISTINCT td.HOUSEHOLD_KEY) as unique_customers FROM transaction_data td LEFT JOIN (SELECT DISTINCT HOUSEHOLD_KEY FROM coupon_redempt) cr ON td.HOUSEHOLD_KEY = cr.HOUSEHOLD_KEY LEFT JOIN (SELECT DISTINCT PRODUCT_ID, STORE_ID, WEEK_NO FROM causal_data WHERE DISPLAY != '0' OR MAILER != '0') cd ON td.PRODUCT_ID = cd.PRODUCT_ID AND td.STORE_ID = cd.STORE_ID AND td.WEEK_NO = cd.WEEK_NO LEFT JOIN (SELECT DISTINCT HOUSEHOLD_KEY FROM campaign_table) ct ON td.HOUSEHOLD_KEY = ct.HOUSEHOLD_KEY GROUP BY attribution_source ORDER BY total_revenue DESC;
"""

def generate_sql(question: str, dataframes: Dict[str, pd.DataFrame], llm: LLM) -> str:
    base_prompt = build_sql_prompt(question, dataframes)
    prompt = f"{base_prompt}\n\n{DATA_DICTIONARY}"

    agent = Agent(
        role="SQL Generator",
        goal="Generate correct SQLite SQL",
        backstory="Only output SQL",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=prompt,
        expected_output="SQL query only",
        agent=agent,
    )

    result = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()
    return clean_sql(str(result))


# ─────────────────────────────────────────────
# INSIGHT GENERATOR (FIXED)
# ─────────────────────────────────────────────

def generate_insight(
    question: str,
    result_df: pd.DataFrame,
    question_type: str,
    llm: LLM,
    kpi_context: Optional[str] = None,
    extra_context: Optional[str] = None,
) -> str:

    # Convert data into strict numbered format
    rows = result_df.head(15)
    numbered_rows = "\n".join(
        f"Row {i+1}: {dict(row)}"
        for i, row in rows.iterrows()
    )

    context_block = f"\nKPI:\n{kpi_context}" if kpi_context else ""
    extra_block = f"\nEXTRA DATA:\n{extra_context}" if extra_context else ""

    # 🎯 STRICT TYPE LOGIC
    type_instruction = {
        "behavioral": """
You are analyzing customers increasing or decreasing spend.

You MUST answer:
1. How many customers are increasing vs decreasing
2. Who are the top customers
3. What categories they buy MOST (Row 1 MUST be used)

DO NOT skip top category.
""",

        "campaign": """
You are analyzing marketing campaigns, coupons, or direct marketing.

CRITICAL:
- If asked a YES/NO question, answer YES or NO first.
- Use exact numbers from the data provided.
- Explain the impact of the campaign based on the metric requested (e.g., engagement, revenue).
""",

        "demographic": """
You are analyzing customer segments.

You MUST:
- Identify top revenue group
- Identify highest avg spend group
- Identify underperforming groups

Use ALL available demographic data.
""",

        "category": """
You are analyzing product categories.

CRITICAL:
- Row 1 is highest revenue
- You MUST start with Row 1
- List top 5 categories with values
""",

        "trend": """
You are analyzing time trends.

You MUST:
- Identify peak and lowest point
- Describe trend direction
"""
        ,
        "custom": """
You are answering a custom analytical question based on a SQL query result.

STRICT RULES for custom:
1. Answer the exact question asked using ONLY the provided DATA rows.
2. Formulate a clear and concise explanation based on the SQL result.
3. Do not assume any standard format unless the data provides it.
4. If the data shows attribution sources (e.g., Organic, Coupon, Campaign), explain the revenue share and what each channel implies for the business."""
    }.get(question_type, "")

    # 🚨 MASTER CONTROL PROMPT (THIS FIXES YOUR ISSUE)
    prompt = f"""
QUESTION: {question}
TYPE: {question_type}

DATA:
{numbered_rows}

{context_block}
{extra_block}

STRICT RULES:
- Row 1 is ALWAYS highest → must be first in answer
- NEVER change ranking
- NEVER invent values
- ONLY use given data
- If YES/NO question → answer YES or NO first
- If you ignore Row 1 → answer is WRONG

RESPONSE FORMAT:

Key Finding:
[direct answer]

Top Insights:
- #1: [Row 1 with value]
- #2: [Row 2]
- #3: [Row 3]

Explanation:
[clear reasoning using actual numbers]

Business Impact:
[what this means]

{type_instruction}
"""

    agent = Agent(
        role="Retail Analyst",
        goal="Give correct insights using real data",
        backstory="You never hallucinate and always follow ranking",
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=prompt,
        expected_output="Accurate structured insight",
        agent=agent,
    )

    result = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()

    return str(result).strip()
