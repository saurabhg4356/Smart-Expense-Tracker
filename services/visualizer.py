"""Matplotlib visualization service for Smart Expense Tracker.

Generates:
1. Category spending pie chart.
2. Monthly spending trend line chart with optional ML prediction forecast.

Charts are saved to the `charts/` directory in headless Agg mode.
Handles empty data gracefully without crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# Set non-interactive backend before importing pyplot to avoid GUI window popups
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models.transaction import Transaction
from services.analytics import monthly_totals, to_dataframe
from services.dsa_utils import category_totals


def plot_category_pie(
    transactions: Sequence[Transaction],
    output_path: Path | str = "charts/category_pie.png",
) -> str | None:
    """Generate and save a pie chart showing spending distribution by category.

    Args:
        transactions: Sequence of transactions.
        output_path: Target image file path.

    Returns:
        Path to saved chart image as string, or None if no expense data exists.
    """
    totals = category_totals(transactions, txn_type="expense")
    if not totals:
        return None

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    categories = list(totals.keys())
    amounts = [totals[c] for c in categories]

    # Harmonious color palette
    colors = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]
    if len(categories) > len(colors):
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    wedges, texts, autotexts = ax.pie(
        amounts,
        labels=categories,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.8,
        textprops={"fontsize": 9},
    )
    for at in autotexts:
        at.set_color("white")
        at.set_weight("bold")

    total_expense = sum(amounts)
    ax.set_title(
        f"Spending by Category (Total: ₹{total_expense:,.2f})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)

    return str(out_file)


def plot_monthly_trend(
    transactions: Sequence[Transaction],
    predicted_next: tuple[float, str] | None = None,
    output_path: Path | str = "charts/monthly_trend.png",
) -> str | None:
    """Generate and save a monthly spending trend line chart with optional ML forecast.

    Args:
        transactions: Sequence of transactions.
        predicted_next: Optional tuple of (predicted_amount, next_month_label).
        output_path: Target image file path.

    Returns:
        Path to saved chart image as string, or None if no expense data exists.
    """
    df = to_dataframe(transactions)
    if df.empty:
        return None

    m_totals = monthly_totals(df)
    if m_totals.empty or m_totals["expense"].sum() == 0:
        return None

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    months = list(m_totals.index)
    expenses = list(m_totals["expense"].values)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Historical spending line
    ax.plot(
        months,
        expenses,
        marker="o",
        linewidth=2.5,
        color="#2B5C8F",
        label="Actual Monthly Spending",
    )

    # Value labels on points
    for m, exp in zip(months, expenses):
        ax.annotate(
            f"₹{exp:,.0f}",
            (m, exp),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            fontweight="semibold",
        )

    # Overlay ML Prediction if provided
    if predicted_next is not None and len(expenses) > 0:
        pred_amount, pred_month = predicted_next
        all_months = months + [pred_month]
        # Line connecting last actual point to predicted point
        ax.plot(
            [months[-1], pred_month],
            [expenses[-1], pred_amount],
            linestyle="--",
            linewidth=2.0,
            color="#E15759",
            label="Forecast Trajectory",
        )
        # Predicted point
        ax.plot(
            [pred_month],
            [pred_amount],
            marker="*",
            markersize=12,
            color="#E15759",
            label=f"Predicted ({pred_month}): ₹{pred_amount:,.0f}",
        )
        ax.annotate(
            f"₹{pred_amount:,.0f}*",
            (pred_month, pred_amount),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=9,
            color="#C82829",
            fontweight="bold",
        )

    ax.set_title("Monthly Spending Trend & Forecast", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=11, labelpad=8)
    ax.set_ylabel("Spending (₹)", fontsize=11, labelpad=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close(fig)

    return str(out_file)
