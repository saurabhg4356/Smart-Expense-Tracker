"""Account hierarchy for Smart Expense Tracker.

Demonstrates Object-Oriented Programming principles:
- Encapsulation of internal financial balance
- Inheritance (SavingsAccount extends Account)
- Polymorphism (method overriding for describe and withdraw)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from models.exceptions import InsufficientFundsError, InvalidAmountError

if TYPE_CHECKING:
    from models.transaction import Transaction


class Account:
    """Standard account maintaining balance with encapsulation.

    Attributes:
        name (str): Identifier name for the account.
        balance (float): Current available balance.
    """

    def __init__(self, name: str, initial_balance: float = 0.0) -> None:
        self._name = (name or "Default").strip()
        self._balance = 0.0
        if initial_balance != 0.0:
            if initial_balance < 0:
                raise InvalidAmountError(f"Initial balance cannot be negative: {initial_balance}")
            self._balance = round(float(initial_balance), 2)

    @property
    def name(self) -> str:
        """Account name."""
        return self._name

    @property
    def balance(self) -> float:
        """Current account balance."""
        return self._balance

    @staticmethod
    def _validate_amount(amount: float) -> float:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise InvalidAmountError(f"Amount must be a numeric value, got: {amount!r}")
        if not math.isfinite(amount):
            raise InvalidAmountError(f"Amount must be finite, got: {amount!r}")
        if amount <= 0:
            raise InvalidAmountError(f"Amount must be strictly positive, got: {amount}")
        return round(float(amount), 2)

    def deposit(self, amount: float) -> None:
        """Deposit funds into account."""
        valid_amount = self._validate_amount(amount)
        self._balance = round(self._balance + valid_amount, 2)

    def withdraw(self, amount: float) -> None:
        """Withdraw funds from account if sufficient balance is present.

        Raises:
            InsufficientFundsError: If requested amount exceeds available balance.
            InvalidAmountError: If amount is non-positive or invalid.
        """
        valid_amount = self._validate_amount(amount)
        if valid_amount > self._balance:
            raise InsufficientFundsError(
                f"Insufficient funds in account '{self._name}': "
                f"requested ₹{valid_amount:,.2f}, available ₹{self._balance:,.2f}."
            )
        self._balance = round(self._balance - valid_amount, 2)

    def apply(self, transaction: Transaction) -> None:
        """Apply a transaction to this account based on its type."""
        if transaction.txn_type == "income":
            self.deposit(transaction.amount)
        elif transaction.txn_type == "expense":
            self.withdraw(transaction.amount)

    def reverse(self, transaction: Transaction) -> None:
        """Reverse the financial effect of a previously applied transaction."""
        if transaction.txn_type == "income":
            self.withdraw(transaction.amount)
        elif transaction.txn_type == "expense":
            self.deposit(transaction.amount)

    def describe(self) -> str:
        """Polymorphic string representation describing account details."""
        return f"Standard Account [{self._name}]: Balance = ₹{self._balance:,.2f}"

    def __repr__(self) -> str:
        return f"Account(name='{self._name}', balance={self._balance:.2f})"


class SavingsAccount(Account):
    """Specialized account enforcing minimum balance and offering interest calculation.

    Demonstrates inheritance, method overriding, and polymorphism.
    """

    def __init__(
        self,
        name: str,
        initial_balance: float = 0.0,
        interest_rate: float = 0.04,
        min_balance: float = 1000.0,
    ) -> None:
        if min_balance < 0:
            raise InvalidAmountError(f"Minimum balance cannot be negative: {min_balance}")
        if interest_rate < 0:
            raise InvalidAmountError(f"Interest rate cannot be negative: {interest_rate}")

        super().__init__(name=name, initial_balance=initial_balance)
        self._interest_rate = float(interest_rate)
        self._min_balance = round(float(min_balance), 2)

    @property
    def interest_rate(self) -> float:
        """Annual interest rate (e.g. 0.04 represents 4%)."""
        return self._interest_rate

    @property
    def min_balance(self) -> float:
        """Required minimum balance."""
        return self._min_balance

    def withdraw(self, amount: float) -> None:
        """Withdraw funds while strictly maintaining required minimum balance.

        Overrides Account.withdraw to enforce the minimum balance constraint.
        """
        valid_amount = self._validate_amount(amount)
        projected_balance = round(self._balance - valid_amount, 2)
        if projected_balance < self._min_balance:
            raise InsufficientFundsError(
                f"Withdrawal of ₹{valid_amount:,.2f} would breach minimum balance requirement "
                f"(₹{self._min_balance:,.2f}) for '{self._name}'. Current balance: ₹{self._balance:,.2f}."
            )
        self._balance = projected_balance

    def monthly_interest(self) -> float:
        """Calculate one month of accrued interest based on current balance."""
        return round((self._balance * self._interest_rate) / 12.0, 2)

    def add_monthly_interest(self) -> float:
        """Credit one month of earned interest into balance and return credited amount."""
        interest = self.monthly_interest()
        if interest > 0:
            self.deposit(interest)
        return interest

    def describe(self) -> str:
        """Overridden polymorphic method displaying savings-specific metrics."""
        return (
            f"Savings Account [{self._name}]: Balance = ₹{self._balance:,.2f} "
            f"(Min Balance = ₹{self._min_balance:,.2f}, APR = {self._interest_rate * 100:.1f}%)"
        )

    def __repr__(self) -> str:
        return (
            f"SavingsAccount(name='{self._name}', balance={self._balance:.2f}, "
            f"min_balance={self._min_balance:.2f}, interest_rate={self._interest_rate:.4f})"
        )
