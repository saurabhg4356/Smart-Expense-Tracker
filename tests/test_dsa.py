"""Unit tests for Data Structures and Algorithms: Hash Map, Heap, Binary Search, and Stack."""

import pytest

from models.transaction import Transaction
from services.dsa_utils import (
    UndoStack,
    build_id_index,
    category_totals,
    top_n_expenses,
    transactions_in_range,
)


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    """Fixture providing sorted transactions with varied types and categories."""
    return [
        Transaction(1, "2026-01-05", 50000.0, "Salary", "Jan Salary", "income", "Bank"),
        Transaction(2, "2026-01-06", 1500.0, "Groceries", "Supermarket", "expense", "Wallet"),
        Transaction(3, "2026-01-10", 350.0, "Food", "Lunch", "expense", "Wallet"),
        Transaction(4, "2026-01-15", 25000.0, "Housing", "Rent", "expense", "Bank"),
        Transaction(5, "2026-01-20", 4200.0, "Shopping", "Winter Coat", "expense", "Wallet"),
        Transaction(6, "2026-01-25", 800.0, "Food", "Dinner", "expense", "Wallet"),
        Transaction(7, "2026-02-01", 50000.0, "Salary", "Feb Salary", "income", "Bank"),
        Transaction(8, "2026-02-05", 3000.0, "Groceries", "Organic Mart", "expense", "Wallet"),
        Transaction(9, "2026-02-14", 12000.0, "Shopping", "Watch", "expense", "Wallet"),
    ]


# -----------------------------------------------------------------------------
# 1. Hash Map: Category Totals & Index Lookup
# -----------------------------------------------------------------------------
def test_category_totals_hash_map(sample_transactions):
    totals = category_totals(sample_transactions, txn_type="expense")

    assert totals["Housing"] == 25000.0
    assert totals["Shopping"] == 16200.0  # 4200 + 12000
    assert totals["Groceries"] == 4500.0  # 1500 + 3000
    assert totals["Food"] == 1150.0       # 350 + 800
    assert "Salary" not in totals         # Ignored income

    income_totals = category_totals(sample_transactions, txn_type="income")
    assert income_totals["Salary"] == 100000.0


def test_build_id_index(sample_transactions):
    idx = build_id_index(sample_transactions)
    assert len(idx) == len(sample_transactions)
    assert idx[4].amount == 25000.0
    assert idx[4].category == "Housing"


# -----------------------------------------------------------------------------
# 2. Heap: Top-N Expenses
# -----------------------------------------------------------------------------
def test_top_n_expenses_heap(sample_transactions):
    # Top 3 expenses should be: Housing (25000), Shopping (12000), Shopping (4200)
    top3 = top_n_expenses(sample_transactions, n=3)
    assert len(top3) == 3
    assert top3[0].amount == 25000.0
    assert top3[1].amount == 12000.0
    assert top3[2].amount == 4200.0


def test_top_n_expenses_ignores_income(sample_transactions):
    # Income of 50,000 must NOT be present in top expenses
    top1 = top_n_expenses(sample_transactions, n=1)
    assert len(top1) == 1
    assert top1[0].amount == 25000.0
    assert top1[0].txn_type == "expense"


def test_top_n_expenses_fewer_items_than_n():
    txns = [
        Transaction(1, "2026-01-01", 100.0, "Food", "Snack", "expense"),
        Transaction(2, "2026-01-02", 200.0, "Food", "Lunch", "expense"),
    ]
    top5 = top_n_expenses(txns, n=5)
    # Should return all available items sorted descending
    assert len(top5) == 2
    assert top5[0].amount == 200.0
    assert top5[1].amount == 100.0


def test_top_n_expenses_empty_and_zero_n():
    assert top_n_expenses([], n=5) == []
    assert top_n_expenses([Transaction(1, "2026-01-01", 100.0, "Food", "Snack", "expense")], n=0) == []


# -----------------------------------------------------------------------------
# 3. Binary Search: Date Range
# -----------------------------------------------------------------------------
def test_transactions_in_range_binary_search(sample_transactions):
    # Query range: 2026-01-10 to 2026-01-25
    results = transactions_in_range(sample_transactions, "2026-01-10", "2026-01-25")
    expected_ids = [3, 4, 5, 6]  # Jan 10, Jan 15, Jan 20, Jan 25
    assert [t.txn_id for t in results] == expected_ids


def test_transactions_in_range_exact_single_day(sample_transactions):
    results = transactions_in_range(sample_transactions, "2026-01-15", "2026-01-15")
    assert len(results) == 1
    assert results[0].txn_id == 4


def test_transactions_in_range_empty():
    txns = [
        Transaction(1, "2026-01-01", 100.0, txn_type="expense"),
        Transaction(2, "2026-01-10", 200.0, txn_type="expense"),
    ]
    # No items in range
    assert transactions_in_range(txns, "2026-01-03", "2026-01-05") == []

    # Inverted range (start > end)
    assert transactions_in_range(txns, "2026-01-10", "2026-01-01") == []


# -----------------------------------------------------------------------------
# 4. Stack: LIFO UndoStack
# -----------------------------------------------------------------------------
def test_undo_stack_lifo():
    stack = UndoStack()
    assert stack.is_empty()
    assert len(stack) == 0

    stack.push("action_1")
    stack.push("action_2")
    stack.push("action_3")

    assert len(stack) == 3
    assert not stack.is_empty()
    assert stack.peek() == "action_3"

    # LIFO pop verification
    assert stack.pop() == "action_3"
    assert stack.pop() == "action_2"
    assert stack.pop() == "action_1"
    assert stack.is_empty()


def test_undo_stack_pop_empty():
    stack = UndoStack()
    with pytest.raises(IndexError):
        stack.pop()
