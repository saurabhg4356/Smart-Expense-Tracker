"""Unit tests for analytics, statistics, anomaly detection, predictor, and categorizer."""

import pytest

from models.exceptions import NotEnoughDataError
from models.transaction import Transaction
from services.analytics import (
    category_breakdown,
    detect_anomalies,
    monthly_totals,
    summary_stats,
    to_dataframe,
)
from services.categorizer import TransactionCategorizer
from services.predictor import SpendingPredictor


@pytest.fixture
def analytics_data() -> list[Transaction]:
    """Diverse 3-month dataset with income and expenses."""
    return [
        # January
        Transaction(1, "2026-01-01", 60000.0, "Salary", "Jan Salary", "income", "Bank"),
        Transaction(2, "2026-01-05", 20000.0, "Housing", "Jan Rent", "expense", "Bank"),
        Transaction(3, "2026-01-10", 2000.0, "Food", "Jan Dining", "expense", "Wallet"),
        Transaction(4, "2026-01-15", 3000.0, "Groceries", "Jan Mart", "expense", "Wallet"),
        # February
        Transaction(5, "2026-02-01", 60000.0, "Salary", "Feb Salary", "income", "Bank"),
        Transaction(6, "2026-02-05", 20000.0, "Housing", "Feb Rent", "expense", "Bank"),
        Transaction(7, "2026-02-12", 2500.0, "Food", "Feb Dining", "expense", "Wallet"),
        Transaction(8, "2026-02-18", 3500.0, "Groceries", "Feb Mart", "expense", "Wallet"),
        # March
        Transaction(9, "2026-03-01", 60000.0, "Salary", "Mar Salary", "income", "Bank"),
        Transaction(10, "2026-03-05", 20000.0, "Housing", "Mar Rent", "expense", "Bank"),
        Transaction(11, "2026-03-10", 2200.0, "Food", "Mar Dining", "expense", "Wallet"),
        Transaction(12, "2026-03-15", 3200.0, "Groceries", "Mar Mart", "expense", "Wallet"),
    ]


# -----------------------------------------------------------------------------
# 1. Summary Statistics & Monthly Grouping
# -----------------------------------------------------------------------------
def test_summary_stats(analytics_data):
    stats = summary_stats(analytics_data)

    assert stats["transaction_count"] == 12
    assert stats["total_income"] == 180000.0
    # Jan expenses: 20000 + 2000 + 3000 = 25000
    # Feb expenses: 20000 + 2500 + 3500 = 26000
    # Mar expenses: 20000 + 2200 + 3200 = 25400
    # Total expenses = 76400
    assert stats["total_expenses"] == 76400.0
    assert stats["net_savings"] == 180000.0 - 76400.0
    assert stats["savings_rate"] > 0
    assert stats["largest_expense"] == 20000.0
    assert stats["average_expense"] > 0
    assert stats["median_expense"] > 0
    assert stats["highest_spending_day"] in ["2026-01-05", "2026-02-05", "2026-03-05"]


def test_monthly_totals_pivoting(analytics_data):
    df = to_dataframe(analytics_data)
    m_totals = monthly_totals(df)

    assert "2026-01" in m_totals.index
    assert "2026-02" in m_totals.index
    assert "2026-03" in m_totals.index

    assert m_totals.loc["2026-01", "income"] == 60000.0
    assert m_totals.loc["2026-01", "expense"] == 25000.0
    assert m_totals.loc["2026-01", "net_savings"] == 35000.0


def test_category_breakdown_sorting(analytics_data):
    df = to_dataframe(analytics_data)
    breakdown = category_breakdown(df, txn_type="expense")

    # Housing is largest expense category (60,000 total)
    assert breakdown.index[0] == "Housing"
    assert breakdown.loc["Housing", "total_amount"] == 60000.0
    assert breakdown.loc["Housing", "count"] == 3
    assert breakdown.loc["Housing", "percentage"] > 50.0


def test_analytics_empty_data():
    empty_stats = summary_stats([])
    assert empty_stats["transaction_count"] == 0
    assert empty_stats["total_income"] == 0.0
    assert empty_stats["total_expenses"] == 0.0
    assert empty_stats["highest_spending_day"] == "N/A"

    df = to_dataframe([])
    assert monthly_totals(df).empty
    assert category_breakdown(df).empty


# -----------------------------------------------------------------------------
# 2. Z-Score Anomaly Detection
# -----------------------------------------------------------------------------
def test_detect_anomalies_z_score():
    # 5 normal shopping expenses, and 1 massive outlier
    txns = [
        Transaction(1, "2026-01-01", 1000.0, "Shopping", "Shirt", "expense"),
        Transaction(2, "2026-01-05", 1200.0, "Shopping", "Pants", "expense"),
        Transaction(3, "2026-01-10", 1100.0, "Shopping", "Shoes", "expense"),
        Transaction(4, "2026-01-15", 950.0, "Shopping", "Hat", "expense"),
        Transaction(5, "2026-01-20", 1050.0, "Shopping", "Belt", "expense"),
        Transaction(6, "2026-01-25", 25000.0, "Shopping", "Diamond Ring", "expense"),
    ]
    anomalies = detect_anomalies(txns, threshold=2.0)
    assert len(anomalies) == 1
    assert anomalies[0]["txn_id"] == 6
    assert anomalies[0]["amount"] == 25000.0
    assert anomalies[0]["z_score"] > 2.0


def test_detect_anomalies_zero_variance():
    # Identical transactions have std = 0, must not throw ZeroDivisionError
    txns = [
        Transaction(1, "2026-01-01", 500.0, "Food", "Meal 1", "expense"),
        Transaction(2, "2026-01-02", 500.0, "Food", "Meal 2", "expense"),
        Transaction(3, "2026-01-03", 500.0, "Food", "Meal 3", "expense"),
    ]
    anomalies = detect_anomalies(txns, threshold=2.0)
    assert anomalies == []


# -----------------------------------------------------------------------------
# 3. Spending Predictor (Linear Regression)
# -----------------------------------------------------------------------------
def test_spending_predictor_insufficient_data():
    predictor = SpendingPredictor()
    # 2 months is fewer than the 3 minimum required
    with pytest.raises(NotEnoughDataError):
        predictor.fit([25000.0, 26000.0])


def test_spending_predictor_fit_and_predict():
    predictor = SpendingPredictor()
    # 4 consecutive months with upward trend: 20000, 22000, 24000, 26000
    monthly = {"2026-01": 20000.0, "2026-02": 22000.0, "2026-03": 24000.0, "2026-04": 26000.0}
    predictor.fit(monthly)

    predicted_amt, mae, next_month = predictor.predict_next()
    # Expected next month spending is ~28000.0, next month label is 2026-05
    assert 27900.0 <= predicted_amt <= 28100.0
    assert next_month == "2026-05"
    assert mae is not None
    assert mae >= 0.0


# -----------------------------------------------------------------------------
# 4. Auto-Categorizer (TF-IDF + Naive Bayes)
# -----------------------------------------------------------------------------
def test_categorizer_predictions():
    categorizer = TransactionCategorizer()

    # Clear food order
    cat_food, conf_food = categorizer.predict_with_confidence("Swiggy biryani dinner order")
    assert cat_food == "Food"
    assert conf_food >= 0.35

    # Clear transport
    cat_trans = categorizer.predict("Uber cab ride to office")
    assert cat_trans == "Transport"

    # Ambiguous or unknown words fall back to 'Other' if below confidence threshold
    cat_unknown, _ = categorizer.predict_with_confidence("xyzqwr random string 999")
    assert cat_unknown == "Other"


# -----------------------------------------------------------------------------
# 5. Visualizer Tests
# -----------------------------------------------------------------------------
def test_visualizer_generate_charts(analytics_data, tmp_path):
    from services.visualizer import plot_category_pie, plot_monthly_trend

    pie_file = tmp_path / "pie.png"
    trend_file = tmp_path / "trend.png"

    res_pie = plot_category_pie(analytics_data, output_path=pie_file)
    assert res_pie is not None
    assert pie_file.is_file()

    res_trend = plot_monthly_trend(
        analytics_data, predicted_next=(27000.0, "2026-04"), output_path=trend_file
    )
    assert res_trend is not None
    assert trend_file.is_file()


def test_visualizer_empty_data(tmp_path):
    from services.visualizer import plot_category_pie, plot_monthly_trend

    # Empty list must return None without raising exception
    assert plot_category_pie([], output_path=tmp_path / "pie.png") is None
    assert plot_monthly_trend([], output_path=tmp_path / "trend.png") is None

