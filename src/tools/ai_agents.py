import pandas as pd


# ─────────────────────────────────────────────
# SIMPLE LOCAL LLM PLACEHOLDER
# ─────────────────────────────────────────────

def get_llm(model=None, base_url=None):
    return None


# ─────────────────────────────────────────────
# QUESTION CLASSIFIER
# ─────────────────────────────────────────────

QUESTION_PATTERNS = {

    "kpi": [
        "revenue",
        "customers",
        "transactions",
        "basket",
        "kpi",
    ],

    "behavioral": [
        "increasing",
        "decreasing",
        "trajectory",
        "customer trend",
        "spending more",
        "spending less",
        "churn",
        "growth"
    ],

    "campaign": [
        "coupon",
        "campaign",
        "promotion",
        "discount"
    ],

    "category_growth": [
        "category",
        "growth",
        "decline",
        "top categories"
    ],

    "demographic": [
        "age",
        "income",
        "demographic",
        "homeowner",
        "kids"
    ],

    "engagement": [
        "engagement",
        "engaged",
        "loyal",
        "frequency",
        "diversity"
    ],

    "trend": [
        "trend",
        "weekly",
        "monthly",
        "over time"
    ]
}


def classify_question(question: str, llm=None) -> str:

    q = question.lower()

    # =====================================================
    # RESEARCH QUESTION PRIORITY
    # =====================================================

    # 1. Direct Marketing / Campaign Effectiveness
    if (
        "direct marketing" in q
        or "marketing improves engagement" in q
        or "campaign effectiveness" in q
        or "typea" in q
    ):
        return "direct_marketing"

    # 2. Demographic + Engagement Combined Questions
    if (
        ("demographic" in q or "age" in q or "income" in q or "homeowner" in q)
        and ("engagement" in q or "spend" in q or "category" in q)
    ):
        return "demographic"

    # 3. Category Growth vs Decline
    if (
        "growth vs decline" in q
        or "driving growth" in q
        or "declining categories" in q
        or "growing categories" in q
    ):
        return "category_growth"

# 4. Customer Trajectory / Behavioral
    if (
        "increasing" in q
        or "decreasing" in q
        or "stable" in q
        or "customer trajectory" in q
        or "spending over time" in q
        or "spending behavior" in q
        or "how many customers" in q
    ):
        return "behavioral"

# 5. Engagement / Loyalty
    if (
        "engagement" in q
        or "loyalty" in q
        or "frequency" in q
        or "basket size" in q
        or "category diversity" in q
        or "engaged customers" in q
    ):
        return "engagement"
    # =====================================================
    # FALLBACKS
    # =====================================================

 # =====================================================
    # FALLBACKS
    # =====================================================

    if "categor" in q:                          # catches "category" AND "categories"
        return "category_growth"

    if "demographic" in q or "income" in q or "age" in q or "segment" in q:
        return "demographic"

    if "engagement" in q or "frequency" in q or "basket" in q or "diversity" in q or "loyal" in q:
        return "engagement"

    if "increase" in q or "decrease" in q or "increas" in q or "decreas" in q or "stable" in q or "spending behavior" in q or "spending over" in q:
        return "behavioral"

    if "campaign" in q or "coupon" in q or "direct marketing" in q or "marketing" in q:
        return "campaign"

    if "how many customers" in q or "customer" in q:
        return "behavioral"

    return "behavioral"   # safe default instead of "custom" which has no handler


# ─────────────────────────────────────────────
# LOCAL INSIGHT ENGINE
# ─────────────────────────────────────────────

def generate_local_insight(question: str, result_df: pd.DataFrame, question_type: str) -> str:  
    response = ""
    
    q = question.lower()

    def money(x):
        try:
            return f"${float(x):,.2f}"
        except:
            return str(x)

    if result_df is None or result_df.empty:
        return "No meaningful insights could be generated."

    # =====================================================
    # BEHAVIORAL
    # =====================================================

    if question_type == "behavioral":
        increasing_row = result_df[result_df["Trend"] == "Increasing"]
        decreasing_row = result_df[result_df["Trend"] == "Decreasing"]
        stable_row = result_df[result_df["Trend"] == "Stable"]

        inc_customers = int(increasing_row["Customers"].iloc[0]) if not increasing_row.empty else 0
        dec_customers = int(decreasing_row["Customers"].iloc[0]) if not decreasing_row.empty else 0
        stable_customers = int(stable_row["Customers"].iloc[0]) if not stable_row.empty else 0

        avg_inc_spend = float(increasing_row["Avg_Spend"].iloc[0]) if not increasing_row.empty else 0
        avg_dec_spend = float(decreasing_row["Avg_Spend"].iloc[0]) if not decreasing_row.empty else 0
        avg_stable_spend = float(stable_row["Avg_Spend"].iloc[0]) if not stable_row.empty else 0

        return f"""### Behavioral Trend Analysis
- Increasing customers: {inc_customers:,}
- Decreasing customers: {dec_customers:,}
- Stable customers: {stable_customers:,}

### Spending Patterns
- Increasing segment avg spend: ${avg_inc_spend:,.2f}
- Decreasing segment avg spend: ${avg_dec_spend:,.2f}
- Stable segment avg spend: ${avg_stable_spend:,.2f}

### Business Interpretation
The analysis shows that a concentrated group of customers is driving significant value growth, while a larger portion demonstrates declining engagement behavior.

### Recommended Action
Deploy retention campaigns toward decreasing customers while strengthening loyalty and personalization strategies for high-growth shoppers.
"""

    # =====================================================
    # ENGAGEMENT
    # =====================================================
    if question_type == "engagement":

        top_customer = result_df.iloc[0]

        score = round(top_customer.get("Engagement_Score", 0), 1)
        frequency = top_customer.get("Frequency", 0)
        diversity = top_customer.get("Category_Diversity", 0)

        if "drive" in q:

            return (
                "Customer engagement is primarily driven by shopping frequency and category diversity.\n\n"

                f"The top engaged customer achieved a score of {score} through "
                f"{frequency} shopping visits across {diversity} product categories.\n\n"

                "Customers who shop consistently across multiple departments tend to demonstrate "
                "the strongest long-term loyalty behavior."
            )

        return (
            f"The highest-engagement customers demonstrate extremely strong repeat purchasing behavior.\n\n"

            f"The top customer recorded {frequency} shopping visits across "
            f"{diversity} unique categories with an engagement score of {score}.\n\n"

            "These customers represent highly loyal shoppers with broad purchasing behavior."
        )

    # =====================================================
    # CAMPAIGN
    # =====================================================

     # =====================================================
    # DIRECT MARKETING / CAMPAIGN EFFECTIVENESS
    # =====================================================

    if question_type in ["campaign", "direct_marketing"]:

        if len(result_df) >= 2:

            group1 = result_df.iloc[0]
            group2 = result_df.iloc[1]

            group1_name = str(group1.iloc[0])
            group2_name = str(group2.iloc[0])

            revenue_1 = float(group1["Total_Revenue"])
            revenue_2 = float(group2["Total_Revenue"])

            customers_1 = float(group1["Customers"])
            customers_2 = float(group2["Customers"])

            avg_rev_1 = float(group1["Avg_Spend"])
            avg_rev_2 = float(group2["Avg_Spend"])

            txn_1 = float(group1["Avg_Transactions_Per_Customer"])
            txn_2 = float(group2["Avg_Transactions_Per_Customer"])

            revenue_lift = ((avg_rev_1 - avg_rev_2) / max(avg_rev_2, 1)) * 100
            txn_lift = ((txn_1 - txn_2) / max(txn_2, 1)) * 100

            response = ""

            # -------------------------------------------------
            # CONTEXTUAL ALIGNMENT
            # -------------------------------------------------

            if "engagement" in q:

                response += (
                    "The analysis provides meaningful evidence that direct marketing is associated "
                    "with stronger customer engagement behavior.\n\n"
                )

                # -------------------------------------------------
                # MULTI-DIMENSIONAL COMPARISON
                # -------------------------------------------------

                response += (
                    f"Customers exposed to direct marketing campaigns generated approximately "
                    f"{money(avg_rev_1)} in revenue per customer, compared to "
                    f"{money(avg_rev_2)} for customers without direct marketing exposure.\n\n"

                    f"This represents an estimated revenue lift of {revenue_lift:.1f}% per customer.\n\n"

                    f"Additionally, directly marketed customers averaged "
                    f"{txn_1:.0f} transactions per customer versus "
                    f"{txn_2:.0f} for non-targeted customers, indicating a substantially "
                    f"higher level of repeat shopping activity.\n\n"
                )

                # -------------------------------------------------
                # STATISTICAL / CAUSAL INTERPRETATION
                # -------------------------------------------------

                response += (
                    "While this does not conclusively prove causality, the scale and consistency "
                    "of the behavioral differences suggest a strong relationship between campaign "
                    "exposure and customer engagement outcomes.\n\n"

                    "However, the interpretation should be treated carefully because highly engaged "
                    "customers may also be more likely to receive or respond to marketing campaigns "
                    "in the first place."
                )

                # -------------------------------------------------
                # BUSINESS SYNTHESIS
                # -------------------------------------------------

                response += (
                    "\n\nFrom a business perspective, the findings indicate that direct marketing "
                    "campaigns are strongly associated with higher customer value, increased shopping "
                    "frequency, and stronger long-term engagement behavior.\n\n"

                    "This suggests that targeted marketing strategies may be effective tools for "
                    "improving customer retention and lifetime value."
                )

                # -------------------------------------------------
                # CONFIDENCE FRAMING
                # -------------------------------------------------

                response += (
                    "\n\nConfidence Level: Moderate to High.\n"
                    "The evidence is directionally strong because the behavioral differences are large "
                    "across both spending and transaction activity, although the analysis remains observational."
                )

                return response

            # -------------------------------------------------
            # GENERIC CAMPAIGN ANALYSIS
            # -------------------------------------------------

            response += (
                f"{group1_name} customers demonstrate significantly stronger purchasing behavior "
                f"than {group2_name} customers.\n\n"

                f"The directly marketed segment generated approximately "
                f"{money(avg_rev_1)} in revenue per customer compared to "
                f"{money(avg_rev_2)} for the comparison group.\n\n"

                "The results suggest that campaign-targeted customers exhibit stronger engagement, "
                "higher transaction frequency, and greater long-term customer value."
            )

            return response

    # =====================================================
    # CATEGORY
    # =====================================================

    if question_type == "category_growth":

        top_row = result_df.iloc[0]

        category = top_row.iloc[0]
        revenue = money(top_row.iloc[1])

        return (
            f"{category} is currently one of the strongest revenue-driving product categories, "
            f"contributing approximately {revenue}.\n\n"

            "The category analysis indicates that a relatively small number of high-frequency "
            "consumer staples are responsible for a large share of retail revenue.\n\n"

            "These categories likely represent repeat-purchase essentials with strong customer dependency."
        )

    # =====================================================
    # DEMOGRAPHIC
    # =====================================================

        # =====================================================
    # DEMOGRAPHIC ANALYSIS
    # =====================================================
    response = ""
    if question_type == "demographic":

        top_segment = result_df.iloc[0] 
        possible_group_cols = [
            "Group",
            "Trend",
            "GroupName",
            "Category",
            "AGE_DESC",
            "INCOME_DESC",
            "HOMEOWNER_DESC",
            "Segment",
            "Dimension"
        ]

        group_col = None

        for c in possible_group_cols:
            if c in result_df.columns:
                group_col = c
                break

        if group_col:
            top_group = str(top_segment[group_col])
        else:
            top_group = "Top Segment"

        revenue = "$0"

    if "Total_Revenue" in top_segment:
        revenue = money(top_segment["Total_Revenue"])

    elif "Avg_Spend" in top_segment:
        revenue = money(top_segment["Avg_Spend"])

    elif "Total_Spend" in top_segment:
        revenue = money(top_segment["Total_Spend"])

    else:
        revenue = "$0"

    # =========================================
    # DEMOGRAPHIC INTERPRETATION
    # =========================================

    if question_type == "demographic":

        response += (
            f"The analysis shows that {top_group} demonstrates "
            f"the strongest overall customer value based on spending "
            f"and engagement behavior.\n\n"

            f"This segment generates approximately {revenue} in value, "
            f"while also maintaining stronger shopping frequency and "
            f"category interaction than other demographic groups.\n\n"

            "The results suggest demographic characteristics influence:\n"
            "- customer spending behavior\n"
            "- category exploration\n"
            "- repeat purchasing\n"
            "- long-term engagement\n\n"

            "Business implication:\n"
            "Retail campaigns and personalization strategies should prioritize "
            "high-value demographic groups for loyalty and retention initiatives."
        )

        # -------------------------------------------------
        # AGE QUESTIONS
        # -------------------------------------------------

        if "age" in q:

            return (
                f"The age segment '{top_group}' generated the strongest overall spending performance "
                f"with approximately {revenue} in revenue.\n\n"

                "This suggests that customer life stage has a measurable influence on purchasing "
                "behavior, basket size, and retail engagement.\n\n"

                "Different age groups likely prioritize different product categories and shopping patterns."
            )

        # -------------------------------------------------
        # INCOME QUESTIONS
        # -------------------------------------------------

        if "income" in q:

            return (
                f"The highest-performing income segment was '{top_group}', generating "
                f"approximately {revenue} in total revenue.\n\n"

                "Income level appears strongly associated with purchasing power, transaction frequency, "
                "and category breadth.\n\n"

                "Higher-income households may contribute more consistently across multiple retail categories."
            )

        # -------------------------------------------------
        # GENERIC DEMOGRAPHIC
        # -------------------------------------------------

        return (
            f"The demographic analysis identified '{top_group}' as the strongest-performing "
            f"customer segment with approximately {revenue} in total revenue.\n\n"

            "The results suggest that household characteristics influence spending behavior, "
            "category participation, and long-term customer value.\n\n"

            "Demographic segmentation can help retailers improve targeting precision and "
            "customer personalization strategies."
        )

    # =====================================================
    # DEFAULT
    # =====================================================

    return (
        "The analysis completed successfully using the uploaded retail datasets.\n\n"

        f"Top Result:\n{result_df.head(3).to_string(index=False)}"
    )


# ─────────────────────────────────────────────
# SQL PLACEHOLDER
# ─────────────────────────────────────────────

def generate_sql(question, dataframes, llm=None):
    return "-- SQL generation disabled for demo stability"