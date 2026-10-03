"""Services package for Smart Expense Tracker."""

from services.analytics import (
    category_breakdown,
    detect_anomalies,
    expenses_only,
    monthly_totals,
    summary_stats,
    to_dataframe,
)
from services.categorizer import TransactionCategorizer
from services.dsa_utils import (
    UndoStack,
    build_id_index,
    category_totals,
    top_n_expenses,
    transactions_in_range,
)
from services.expense_manager import ExpenseManager
from services.predictor import SpendingPredictor
from services.storage import load_transactions, save_transactions
from services.visualizer import plot_category_pie, plot_monthly_trend

__all__ = [
    "load_transactions",
    "save_transactions",
    "category_totals",
    "build_id_index",
    "top_n_expenses",
    "transactions_in_range",
    "UndoStack",
    "to_dataframe",
    "expenses_only",
    "monthly_totals",
    "category_breakdown",
    "summary_stats",
    "detect_anomalies",
    "SpendingPredictor",
    "TransactionCategorizer",
    "plot_category_pie",
    "plot_monthly_trend",
    "ExpenseManager",
]
