"""Unit tests for ExpenseManager service coordinating CRUD, balances, undo, and persistence."""

from pathlib import Path

import pytest

from models.exceptions import (
    InsufficientFundsError,
    TransactionNotFoundError,
)
from services.expense_manager import ExpenseManager


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Fixture providing a temporary CSV file path for isolated tests."""
    return tmp_path / "test_transactions.csv"


def test_add_transaction_updates_balance(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)

    # Deposit income first
    mgr.add_transaction(
        txn_date="2026-04-01",
        amount=10000.0,
        category="Salary",
        txn_type="income",
        account="Wallet",
    )
    assert mgr.accounts["Wallet"].balance == 10000.0

    # Add expense
    mgr.add_transaction(
        txn_date="2026-04-02",
        amount=2500.0,
        category="Groceries",
        txn_type="expense",
        account="Wallet",
    )
    assert mgr.accounts["Wallet"].balance == 7500.0


def test_expense_before_income_fails_insufficient_funds(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    # Wallet balance starts at 0.0
    with pytest.raises(InsufficientFundsError):
        mgr.add_transaction(
            txn_date="2026-04-01",
            amount=500.0,
            category="Food",
            txn_type="expense",
            account="Wallet",
        )
    # State should remain empty
    assert len(mgr.get_all_transactions()) == 0


def test_transactions_maintained_sorted(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    # Seed funds
    mgr.add_transaction("2026-01-01", 50000.0, txn_type="income", account="Bank")

    # Add transactions in random date order
    mgr.add_transaction("2026-03-10", 500.0, txn_type="expense", account="Bank")
    mgr.add_transaction("2026-01-15", 300.0, txn_type="expense", account="Bank")
    mgr.add_transaction("2026-02-20", 400.0, txn_type="expense", account="Bank")

    all_txns = mgr.get_all_transactions()
    dates = [t.txn_date for t in all_txns]
    assert dates == sorted(dates)


def test_delete_transaction_reverses_balance(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    mgr.add_transaction("2026-01-01", 10000.0, txn_type="income", account="Wallet")
    exp = mgr.add_transaction("2026-01-02", 3000.0, txn_type="expense", account="Wallet")
    assert mgr.accounts["Wallet"].balance == 7000.0

    # Delete expense
    mgr.delete_transaction(exp.txn_id)
    assert mgr.accounts["Wallet"].balance == 10000.0
    assert len(mgr.get_all_transactions()) == 1


def test_delete_nonexistent_transaction(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    with pytest.raises(TransactionNotFoundError):
        mgr.delete_transaction(9999)


def test_update_transaction_adjusts_balance(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    mgr.add_transaction("2026-01-01", 10000.0, txn_type="income", account="Wallet")
    exp = mgr.add_transaction("2026-01-02", 2000.0, txn_type="expense", account="Wallet")
    assert mgr.accounts["Wallet"].balance == 8000.0

    # Update amount from 2000 to 3500
    mgr.update_transaction(exp.txn_id, amount=3500.0)
    assert mgr.accounts["Wallet"].balance == 6500.0

    updated = mgr.get_transaction(exp.txn_id)
    assert updated.amount == 3500.0


def test_undo_add_delete_update(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)

    # Test undo on empty
    assert mgr.undo() == "Nothing to undo."

    # 1. Undo ADD
    mgr.add_transaction("2026-01-01", 5000.0, txn_type="income", account="Wallet")
    assert mgr.accounts["Wallet"].balance == 5000.0
    msg = mgr.undo()
    assert "Undid ADD" in msg
    assert mgr.accounts["Wallet"].balance == 0.0
    assert len(mgr.get_all_transactions()) == 0

    # 2. Undo DELETE
    inc = mgr.add_transaction("2026-01-01", 5000.0, txn_type="income", account="Wallet")
    mgr.delete_transaction(inc.txn_id)
    assert len(mgr.get_all_transactions()) == 0
    msg2 = mgr.undo()
    assert "Undid DELETE" in msg2
    assert len(mgr.get_all_transactions()) == 1
    assert mgr.accounts["Wallet"].balance == 5000.0

    # 3. Undo UPDATE
    exp = mgr.add_transaction("2026-01-02", 1000.0, txn_type="expense", account="Wallet")
    assert mgr.accounts["Wallet"].balance == 4000.0
    mgr.update_transaction(exp.txn_id, amount=2500.0)
    assert mgr.accounts["Wallet"].balance == 2500.0
    msg3 = mgr.undo()
    assert "Undid UPDATE" in msg3
    assert mgr.get_transaction(exp.txn_id).amount == 1000.0
    assert mgr.accounts["Wallet"].balance == 4000.0


def test_persistence_save_and_reload(tmp_storage):
    mgr1 = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    mgr1.add_transaction("2026-01-01", 20000.0, txn_type="income", account="Bank")
    mgr1.add_transaction("2026-01-05", 4500.0, category="Groceries", txn_type="expense", account="Bank")

    # Reload fresh manager instance from the same file
    mgr2 = ExpenseManager(storage_path=tmp_storage, auto_load=True)
    all_txns = mgr2.get_all_transactions()
    assert len(all_txns) == 2
    assert mgr2.accounts["Bank"].balance == 15500.0
    assert all_txns[1].category == "Groceries"


def test_search_transactions(tmp_storage):
    mgr = ExpenseManager(storage_path=tmp_storage, auto_load=False)
    mgr.add_transaction("2026-01-01", 20000.0, category="Salary", description="Tech Corp", txn_type="income")
    mgr.add_transaction("2026-01-05", 650.0, category="Food", description="Swiggy dinner", txn_type="expense")

    # Query matches description
    results1 = mgr.search_transactions("swiggy")
    assert len(results1) == 1
    assert results1[0].category == "Food"

    # Query matches category
    results2 = mgr.search_transactions("Salary")
    assert len(results2) == 1

    # Empty search query
    assert mgr.search_transactions("") == []


def test_corrupted_csv_handling(tmp_path):
    from models.exceptions import StorageError
    from services.storage import load_transactions

    # Missing required header
    bad_csv = tmp_path / "bad_header.csv"
    bad_csv.write_text("random,headers,here\n1,2,3\n", encoding="utf-8")
    with pytest.raises(StorageError):
        load_transactions(bad_csv)

    # Corrupted data row (invalid amount)
    bad_data_csv = tmp_path / "bad_data.csv"
    bad_data_csv.write_text(
        "txn_id,txn_date,amount,category,description,txn_type,account\n"
        "1,2026-01-01,not_a_number,Food,Lunch,expense,Wallet\n",
        encoding="utf-8",
    )
    with pytest.raises(StorageError):
        load_transactions(bad_data_csv)


def test_empty_database_start(tmp_path):
    nonexistent = tmp_path / "nonexistent.csv"
    mgr = ExpenseManager(storage_path=nonexistent, auto_load=True)
    assert len(mgr.get_all_transactions()) == 0
    assert mgr.undo() == "Nothing to undo."

