"""Transaction entity for Smart Expense Tracker.

Demonstrates Object-Oriented Programming principles:
- Encapsulation with validated properties
- Dunder methods for string representation and comparisons
- Classmethod factory for dictionary deserialization
"""

from __future__ import annotations

import datetime
import functools
import math
from typing import Any

from models.exceptions import InvalidAmountError, InvalidTransactionError


@functools.total_ordering
class Transaction:
    """Represents a financial transaction (income or expense).

    Attributes:
        txn_id (int): Unique identifier for the transaction.
        txn_date (str): Transaction date in ISO format (YYYY-MM-DD).
        amount (float): Transaction monetary value (strictly positive, rounded to 2 decimals).
        category (str): Classification category (default: 'Other').
        description (str): Transaction narrative or memo.
        txn_type (str): Type of transaction ('income' or 'expense').
        account (str): Source or destination account (default: 'Wallet').
    """

    VALID_TYPES = ("income", "expense")
    DEFAULT_CATEGORY = "Other"
    DEFAULT_ACCOUNT = "Wallet"

    def __init__(
        self,
        txn_id: int,
        txn_date: str,
        amount: float,
        category: str = DEFAULT_CATEGORY,
        description: str = "",
        txn_type: str = "expense",
        account: str = DEFAULT_ACCOUNT,
    ) -> None:
        self._txn_id = self._validate_id(txn_id)
        self._txn_date = self._validate_date(txn_date)
        self._amount = self._validate_amount(amount)
        self._category = (category or self.DEFAULT_CATEGORY).strip() or self.DEFAULT_CATEGORY
        self._description = (description or "").strip()
        self._txn_type = self._validate_type(txn_type)
        self._account = (account or self.DEFAULT_ACCOUNT).strip() or self.DEFAULT_ACCOUNT

    # -------------------------------------------------------------------------
    # Validation helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_id(val: Any) -> int:
        if isinstance(val, bool) or not isinstance(val, int) or val < 0:
            raise InvalidTransactionError(f"Transaction ID must be a non-negative integer, got: {val!r}")
        return val

    @staticmethod
    def _validate_amount(val: Any) -> float:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise InvalidAmountError(f"Amount must be a numeric value, got: {val!r}")
        if not math.isfinite(val):
            raise InvalidAmountError(f"Amount must be a finite number, got: {val!r}")
        if val <= 0:
            raise InvalidAmountError(f"Amount must be greater than zero, got: {val!r}")
        return round(float(val), 2)

    @staticmethod
    def _validate_type(val: Any) -> str:
        if not isinstance(val, str):
            raise InvalidTransactionError(f"Transaction type must be a string, got: {val!r}")
        cleaned = val.strip().lower()
        if cleaned not in Transaction.VALID_TYPES:
            raise InvalidTransactionError(
                f"Invalid transaction type '{val}'. Must be one of: {Transaction.VALID_TYPES}"
            )
        return cleaned

    @staticmethod
    def _validate_date(val: Any) -> str:
        if not isinstance(val, str):
            raise InvalidTransactionError(f"Transaction date must be an ISO format string (YYYY-MM-DD), got: {val!r}")
        cleaned = val.strip()
        try:
            parsed = datetime.date.fromisoformat(cleaned)
            return parsed.isoformat()
        except ValueError as err:
            raise InvalidTransactionError(
                f"Invalid date format '{val}'. Expected valid ISO format YYYY-MM-DD."
            ) from err

    # -------------------------------------------------------------------------
    # Encapsulated Properties
    # -------------------------------------------------------------------------
    @property
    def txn_id(self) -> int:
        """Unique integer identifier."""
        return self._txn_id

    @txn_id.setter
    def txn_id(self, value: int) -> None:
        self._txn_id = self._validate_id(value)

    @property
    def txn_date(self) -> str:
        """ISO date string (YYYY-MM-DD)."""
        return self._txn_date

    @txn_date.setter
    def txn_date(self, value: str) -> None:
        self._txn_date = self._validate_date(value)

    @property
    def amount(self) -> float:
        """Transaction monetary amount."""
        return self._amount

    @amount.setter
    def amount(self, value: float) -> None:
        self._amount = self._validate_amount(value)

    @property
    def category(self) -> str:
        """Transaction category name."""
        return self._category

    @category.setter
    def category(self, value: str) -> None:
        self._category = (value or self.DEFAULT_CATEGORY).strip() or self.DEFAULT_CATEGORY

    @property
    def description(self) -> str:
        """Transaction memo / description."""
        return self._description

    @description.setter
    def description(self, value: str) -> None:
        self._description = (value or "").strip()

    @property
    def txn_type(self) -> str:
        """'income' or 'expense'."""
        return self._txn_type

    @txn_type.setter
    def txn_type(self, value: str) -> None:
        self._txn_type = self._validate_type(value)

    @property
    def account(self) -> str:
        """Target account name."""
        return self._account

    @account.setter
    def account(self, value: str) -> None:
        self._account = (value or self.DEFAULT_ACCOUNT).strip() or self.DEFAULT_ACCOUNT

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Convert transaction instance to a dictionary representation."""
        return {
            "txn_id": self._txn_id,
            "txn_date": self._txn_date,
            "amount": self._amount,
            "category": self._category,
            "description": self._description,
            "txn_type": self._txn_type,
            "account": self._account,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transaction:
        """Create a Transaction instance from a dictionary.

        Raises:
            InvalidTransactionError: If required fields are missing or malformed.
            InvalidAmountError: If amount is non-numeric or non-positive.
        """
        required = ("txn_id", "txn_date", "amount", "txn_type")
        for key in required:
            if key not in data:
                raise InvalidTransactionError(f"Missing required transaction field: '{key}'")

        try:
            raw_id = int(data["txn_id"])
        except (ValueError, TypeError) as err:
            raise InvalidTransactionError(f"Invalid txn_id: {data.get('txn_id')!r}") from err

        try:
            raw_amount = float(data["amount"])
        except (ValueError, TypeError) as err:
            raise InvalidAmountError(f"Invalid amount value: {data.get('amount')!r}") from err

        return cls(
            txn_id=raw_id,
            txn_date=str(data["txn_date"]),
            amount=raw_amount,
            category=str(data.get("category", cls.DEFAULT_CATEGORY)),
            description=str(data.get("description", "")),
            txn_type=str(data["txn_type"]),
            account=str(data.get("account", cls.DEFAULT_ACCOUNT)),
        )

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Transaction(id={self._txn_id}, date='{self._txn_date}', amount={self._amount:.2f}, "
            f"type='{self._txn_type}', category='{self._category}', account='{self._account}')"
        )

    def __str__(self) -> str:
        sign = "+" if self._txn_type == "income" else "-"
        return (
            f"[{self._txn_id:04d}] {self._txn_date} | {sign}₹{self._amount:,.2f} | "
            f"{self._category:<14} | {self._account:<8} | {self._description}"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Transaction):
            return NotImplemented
        return (
            self._txn_id == other._txn_id
            and self._txn_date == other._txn_date
            and math.isclose(self._amount, other._amount, abs_tol=1e-5)
            and self._category == other._category
            and self._description == other._description
            and self._txn_type == other._txn_type
            and self._account == other._account
        )

    def __lt__(self, other: object) -> bool:
        """Sort primary by txn_date, secondary by txn_id."""
        if not isinstance(other, Transaction):
            return NotImplemented
        return (self._txn_date, self._txn_id) < (other._txn_date, other._txn_id)
