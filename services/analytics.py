"""Financial analytics and anomaly detection service using pandas and NumPy.

Provides data transformation, statistical aggregation, savings rate computation,
and category-specific z-score anomaly detection.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from models.transaction import Transaction


def to_dataframe(transactions: Sequence[Transaction]) -> pd.DataFrame:
    """Convert a sequence of Transaction objects to a pandas DataFrame.

    Returns:
        DataFrame with standardized columns, or empty DataFrame with columns if empty.
    """
    columns = [
        "txn_id",
        "txn_date",
        "amount",
        "category",
        "description",
        "txn_type",
        "account",
    ]
    if not transactions:
        return pd.DataFrame(columns=columns)

    records = [txn.to_dict() for txn in transactions]
    df = pd.DataFrame(records, columns=columns)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["txn_date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    return df


def expenses_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include expense transactions."""
    if df.empty or "txn_type" not in df.columns:
        return df
    return df[df["txn_type"] == "expense"].copy()


def monthly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total income, total expenses, and net savings grouped by month (YYYY-MM).

    Returns:
        DataFrame indexed by Month with columns ['income', 'expense', 'net_savings'].
    """
    if df.empty:
        return pd.DataFrame(columns=["income", "expense", "net_savings"])

    work_df = df.copy()
    work_df["month"] = work_df["txn_date"].dt.to_period("M").astype(str)

    pivoted = (
        work_df.pivot_table(
            index="month",
            columns="txn_type",
            values="amount",
            aggfunc="sum",
            fill_value=0.0,
        )
    )

    if "income" not in pivoted.columns:
        pivoted["income"] = 0.0
    if "expense" not in pivoted.columns:
        pivoted["expense"] = 0.0

    pivoted = pivoted[["income", "expense"]]
    pivoted["net_savings"] = pivoted["income"] - pivoted["expense"]
    return pivoted.sort_index()


def category_breakdown(df: pd.DataFrame, txn_type: str = "expense") -> pd.DataFrame:
    """Compute spending or income breakdown grouped by category.

    Calculates total amount, transaction count, and percentage share.
    """
    if df.empty or "txn_type" not in df.columns:
        return pd.DataFrame(columns=["total_amount", "count", "percentage"])

    filtered = df[df["txn_type"] == txn_type]
    if filtered.empty:
        return pd.DataFrame(columns=["total_amount", "count", "percentage"])

    grouped = filtered.groupby("category")["amount"].agg(["sum", "count"]).rename(
        columns={"sum": "total_amount"}
    )
    total_spent = grouped["total_amount"].sum()
    grouped["percentage"] = (
        (grouped["total_amount"] / total_spent * 100.0) if total_spent > 0 else 0.0
    )
    return grouped.sort_values(by="total_amount", ascending=False)


def summary_stats(transactions: Sequence[Transaction]) -> dict[str, Any]:
    """Calculate comprehensive statistical summary of financial transactions.

    Uses pandas and NumPy for analytical and statistical calculations.
    """
    empty_result = {
        "transaction_count": 0,
        "total_income": 0.0,
        "total_expenses": 0.0,
        "net_savings": 0.0,
        "savings_rate": 0.0,
        "average_expense": 0.0,
        "median_expense": 0.0,
        "std_expense": 0.0,
        "largest_expense": 0.0,
        "highest_spending_day": "N/A",
        "category_breakdown": {},
        "monthly_spending": {},
    }

    if not transactions:
        return empty_result

    df = to_dataframe(transactions)
    if df.empty:
        return empty_result

    incomes = df[df["txn_type"] == "income"]["amount"].to_numpy()
    expenses = df[df["txn_type"] == "expense"]["amount"].to_numpy()

    total_income = float(np.sum(incomes)) if len(incomes) > 0 else 0.0
    total_expenses = float(np.sum(expenses)) if len(expenses) > 0 else 0.0
    net_savings = round(total_income - total_expenses, 2)

    savings_rate = (
        round((net_savings / total_income) * 100.0, 2)
        if total_income > 0
        else 0.0
    )

    if len(expenses) > 0:
        avg_exp = round(float(np.mean(expenses)), 2)
        med_exp = round(float(np.median(expenses)), 2)
        std_exp = round(float(np.std(expenses, ddof=1)), 2) if len(expenses) > 1 else 0.0
        max_exp = round(float(np.max(expenses)), 2)
    else:
        avg_exp = 0.0
        med_exp = 0.0
        std_exp = 0.0
        max_exp = 0.0

    # Highest spending day
    highest_day = "N/A"
    exp_df = df[df["txn_type"] == "expense"]
    if not exp_df.empty:
        day_spending = exp_df.groupby(exp_df["txn_date"].dt.strftime("%Y-%m-%d"))["amount"].sum()
        if not day_spending.empty:
            highest_day = str(day_spending.idxmax())

    # Category breakdown (dict of name -> sum)
    cat_df = category_breakdown(df, txn_type="expense")
    cat_dict = (
        {str(cat): round(float(row["total_amount"]), 2) for cat, row in cat_df.iterrows()}
        if not cat_df.empty
        else {}
    )

    # Monthly spending (dict of YYYY-MM -> expense sum)
    m_totals = monthly_totals(df)
    month_dict = (
        {str(m): round(float(m_totals.loc[m, "expense"]), 2) for m in m_totals.index}
        if not m_totals.empty
        else {}
    )

    return {
        "transaction_count": len(transactions),
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "net_savings": net_savings,
        "savings_rate": savings_rate,
        "average_expense": avg_exp,
        "median_expense": med_exp,
        "std_expense": std_exp,
        "largest_expense": max_exp,
        "highest_spending_day": highest_day,
        "category_breakdown": cat_dict,
        "monthly_spending": month_dict,
    }


def detect_anomalies(
    transactions: Sequence[Transaction], threshold: float = 2.0
) -> list[dict[str, Any]]:
    """Detect statistical spending anomalies using category-based z-scores.

    For each expense category:
        z = (amount - mean) / standard_deviation
    Flag transactions where |z| > threshold.

    Guards against division-by-zero when standard deviation is 0.

    Args:
        transactions: Sequence of Transaction objects.
        threshold: Z-score threshold (default 2.0, representing ~95% confidence).

    Returns:
        List of anomaly records sorted descending by absolute z-score.
    """
    expense_txns = [t for t in transactions if t.txn_type == "expense"]
    if not expense_txns:
        return []

    # Group transactions by category
    by_category: dict[str, list[Transaction]] = {}
    for txn in expense_txns:
        by_category.setdefault(txn.category, []).append(txn)

    anomalies: list[dict[str, Any]] = []

    for cat, txns in by_category.items():
        # Need at least 2 data points to compute sample standard deviation
        if len(txns) < 2:
            continue

        amounts = np.array([t.amount for t in txns], dtype=float)
        mean = float(np.mean(amounts))
        std = float(np.std(amounts, ddof=1))

        # Avoid division by zero if all values in category are identical
        if std <= 1e-6 or np.isnan(std):
            continue

        for txn in txns:
            z_score = (txn.amount - mean) / std
            if abs(z_score) > threshold:
                anomalies.append({
                    "transaction": txn,
                    "txn_id": txn.txn_id,
                    "date": txn.txn_date,
                    "category": cat,
                    "description": txn.description,
                    "amount": txn.amount,
                    "z_score": round(float(z_score), 2),
                    "category_mean": round(mean, 2),
                    "category_std": round(std, 2),
                    "threshold": threshold,
                })

    # Sort anomalies by highest divergence (absolute z-score)
    anomalies.sort(key=lambda item: abs(item["z_score"]), reverse=True)
    return anomalies
