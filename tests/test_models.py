"""Unit tests for models: Transaction, Account, SavingsAccount, and Exceptions."""

import pytest

from models.account import Account, SavingsAccount
from models.exceptions import (
    InsufficientFundsError,
    InvalidAmountError,
    InvalidTransactionError,
)
from models.transaction import Transaction


# -----------------------------------------------------------------------------
# Transaction Tests
# -----------------------------------------------------------------------------
def test_valid_transaction_creation():
    txn = Transaction(
        txn_id=1,
        txn_date="2026-03-15",
        amount=1250.50,
        category="Groceries",
        description="Weekly vegetables",
        txn_type="expense",
        account="Wallet",
    )
    assert txn.txn_id == 1
    assert txn.txn_date == "2026-03-15"
    assert txn.amount == 1250.50
    assert txn.category == "Groceries"
    assert txn.description == "Weekly vegetables"
    assert txn.txn_type == "expense"
    assert txn.account == "Wallet"


def test_transaction_invalid_negative_and_zero_amount():
    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", -50.0)

    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", 0.0)


def test_transaction_invalid_non_numeric_amount():
    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", "five-hundred")  # type: ignore

    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", True)  # bool is not allowed as amount

    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", float("inf"))

    with pytest.raises(InvalidAmountError):
        Transaction(1, "2026-01-01", float("nan"))


def test_transaction_invalid_type():
    with pytest.raises(InvalidTransactionError):
        Transaction(1, "2026-01-01", 100.0, txn_type="investment")


def test_transaction_valid_and_invalid_date():
    # Valid ISO date
    txn = Transaction(1, "2026-02-28", 100.0)
    assert txn.txn_date == "2026-02-28"

    # Invalid dates
    with pytest.raises(InvalidTransactionError):
        Transaction(1, "28-02-2026", 100.0)

    with pytest.raises(InvalidTransactionError):
        Transaction(1, "2026-02-30", 100.0)  # Feb 30 does not exist

    with pytest.raises(InvalidTransactionError):
        Transaction(1, "not-a-date", 100.0)


def test_transaction_ordering():
    t1 = Transaction(2, "2026-01-15", 50.0)
    t2 = Transaction(1, "2026-01-15", 100.0)
    t3 = Transaction(3, "2026-01-10", 20.0)
    t4 = Transaction(4, "2026-02-01", 30.0)

    # Should sort primarily by txn_date, secondarily by txn_id
    sorted_txns = sorted([t1, t2, t3, t4])
    assert sorted_txns == [t3, t2, t1, t4]


def test_transaction_dict_roundtrip():
    original = Transaction(
        txn_id=42,
        txn_date="2026-05-20",
        amount=3499.99,
        category="Shopping",
        description="Noise-cancelling headphones",
        txn_type="expense",
        account="Bank",
    )
    raw_dict = original.to_dict()
    reconstructed = Transaction.from_dict(raw_dict)
    assert original == reconstructed
    assert reconstructed.txn_id == 42
    assert reconstructed.amount == 3499.99


# -----------------------------------------------------------------------------
# Account Tests
# -----------------------------------------------------------------------------
def test_account_deposit_and_withdraw():
    acc = Account("Checking", initial_balance=5000.0)
    assert acc.balance == 5000.0

    acc.deposit(2000.0)
    assert acc.balance == 7000.0

    acc.withdraw(3000.0)
    assert acc.balance == 4000.0


def test_account_insufficient_funds():
    acc = Account("Wallet", initial_balance=500.0)
    with pytest.raises(InsufficientFundsError):
        acc.withdraw(600.0)
    assert acc.balance == 500.0  # Balance unchanged


def test_account_apply_and_reverse():
    acc = Account("Bank", initial_balance=1000.0)
    income_txn = Transaction(1, "2026-01-01", 2000.0, txn_type="income", account="Bank")
    exp_txn = Transaction(2, "2026-01-02", 500.0, txn_type="expense", account="Bank")

    acc.apply(income_txn)
    assert acc.balance == 3000.0

    acc.apply(exp_txn)
    assert acc.balance == 2500.0

    acc.reverse(exp_txn)
    assert acc.balance == 3000.0

    acc.reverse(income_txn)
    assert acc.balance == 1000.0


# -----------------------------------------------------------------------------
# SavingsAccount (Inheritance & Polymorphism) Tests
# -----------------------------------------------------------------------------
def test_savings_account_minimum_balance():
    savings = SavingsAccount("Emergency", initial_balance=5000.0, min_balance=1000.0)

    # Allowed withdrawal: 5000 - 3500 = 1500 (>= 1000)
    savings.withdraw(3500.0)
    assert savings.balance == 1500.0

    # Disallowed withdrawal: 1500 - 600 = 900 (< 1000 min_balance)
    with pytest.raises(InsufficientFundsError) as exc_info:
        savings.withdraw(600.0)
    assert "minimum balance requirement" in str(exc_info.value)
    assert savings.balance == 1500.0


def test_savings_account_interest():
    # Balance 12,000 at 5% APR = (12000 * 0.05) / 12 = 50.00 / month
    savings = SavingsAccount("HighYield", initial_balance=12000.0, interest_rate=0.05)
    monthly_interest = savings.monthly_interest()
    assert monthly_interest == 50.0

    credited = savings.add_monthly_interest()
    assert credited == 50.0
    assert savings.balance == 12050.0


def test_polymorphic_describe():
    acc = Account("General", initial_balance=100.0)
    sav = SavingsAccount("Retirement", initial_balance=5000.0, interest_rate=0.06, min_balance=1000.0)

    # Both have describe() method returning polymorphic string representations
    accounts: list[Account] = [acc, sav]
    descriptions = [a.describe() for a in accounts]

    assert "Standard Account [General]" in descriptions[0]
    assert "Savings Account [Retirement]" in descriptions[1]
    assert "APR = 6.0%" in descriptions[1]
    assert "Min Balance = ₹1,000.00" in descriptions[1]
