"""Main business logic and service layer: ExpenseManager.

Coordinates:
- Transaction management (Add, Delete, Update, Search, Date Range Filter, Top-N)
- Double-entry balance synchronization with Account entities
- LIFO Undo system using UndoStack
- Binary search insertion maintaining sorted order by (txn_date, txn_id)
- Automatic CSV persistence
- Transaction categorization using ML
"""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Sequence

from models.account import Account, SavingsAccount
from models.exceptions import (
    InsufficientFundsError,
    InvalidTransactionError,
    StorageError,
    TransactionNotFoundError,
)
from models.transaction import Transaction
from services.categorizer import TransactionCategorizer
from services.dsa_utils import UndoStack, build_id_index, top_n_expenses, transactions_in_range
from services.storage import load_transactions, save_transactions


class ExpenseManager:
    """Core domain service managing transactions, accounts, undo stack, and persistence."""

    DEFAULT_STORAGE = Path("data/transactions.csv")

    def __init__(
        self,
        storage_path: Path | str = DEFAULT_STORAGE,
        auto_load: bool = True,
    ) -> None:
        self.storage_path = Path(storage_path)
        self._transactions: list[Transaction] = []
        self._by_id: dict[int, Transaction] = {}
        self._undo = UndoStack()
        self.accounts: dict[str, Account] = {}
        self.categorizer = TransactionCategorizer()
        self._next_id: int = 1

        self._initialize_default_accounts()

        if auto_load and self.storage_path.is_file():
            self.load()

    def _initialize_default_accounts(self) -> None:
        """Create baseline financial accounts."""
        self.accounts = {
            "Wallet": Account("Wallet", initial_balance=0.0),
            "Bank": Account("Bank", initial_balance=0.0),
            "Savings": SavingsAccount(
                "Savings",
                initial_balance=0.0,
                interest_rate=0.04,
                min_balance=1000.0,
            ),
        }

    def get_or_create_account(self, account_name: str) -> Account:
        """Retrieve existing account or create a new Account dynamically."""
        name = (account_name or "Wallet").strip()
        if name in self.accounts:
            return self.accounts[name]

        if "saving" in name.lower():
            acc = SavingsAccount(name, initial_balance=0.0, min_balance=1000.0)
        else:
            acc = Account(name, initial_balance=0.0)
        self.accounts[name] = acc
        return acc

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def load(self, custom_path: Path | str | None = None) -> None:
        """Load transactions from CSV storage and replay into account balances."""
        target = Path(custom_path) if custom_path else self.storage_path
        if not target.is_file():
            return

        loaded = load_transactions(target)
        # Re-sort to guarantee (txn_date, txn_id) order
        loaded.sort()

        self._transactions = []
        self._by_id.clear()
        self._undo.clear()
        self._initialize_default_accounts()

        max_id = 0
        for txn in loaded:
            if txn.txn_id > max_id:
                max_id = txn.txn_id

            acc = self.get_or_create_account(txn.account)
            try:
                acc.apply(txn)
            except InsufficientFundsError:
                # If historical data has a negative balance prior to deposit,
                # seed initial balance to allow loading without loss of fidelity.
                deficit = txn.amount - acc.balance
                acc.deposit(deficit)
                acc.apply(txn)

            self._transactions.append(txn)
            self._by_id[txn.txn_id] = txn

        self._next_id = max_id + 1
        # Retrain categorizer with historical user patterns
        self.categorizer.fit_from_transactions(self._transactions)

    def save(self, custom_path: Path | str | None = None) -> None:
        """Persist all transactions to CSV storage."""
        target = Path(custom_path) if custom_path else self.storage_path
        save_transactions(self._transactions, target)

    # -------------------------------------------------------------------------
    # Transaction CRUD Operations
    # -------------------------------------------------------------------------
    def add_transaction(
        self,
        txn_date: str,
        amount: float,
        category: str = "Other",
        description: str = "",
        txn_type: str = "expense",
        account: str = "Wallet",
        auto_categorize: bool = False,
    ) -> Transaction:
        """Add a new transaction, adjust account balance, update index, and persist.

        Raises:
            InsufficientFundsError: If an expense exceeds available account funds.
            InvalidAmountError: If amount is non-numeric or non-positive.
            InvalidTransactionError: If date or type fails validation.
        """
        final_category = category
        if auto_categorize or not final_category or final_category.strip().lower() == "other":
            predicted, _ = self.categorizer.predict_with_confidence(description)
            if predicted != "Other":
                final_category = predicted

        acc = self.get_or_create_account(account)

        txn = Transaction(
            txn_id=self._next_id,
            txn_date=txn_date,
            amount=amount,
            category=final_category,
            description=description,
            txn_type=txn_type,
            account=acc.name,
        )

        # Apply transaction to account; raises InsufficientFundsError before modifying state
        acc.apply(txn)

        # Maintain sorted order by (txn_date, txn_id)
        bisect.insort(self._transactions, txn)
        self._by_id[txn.txn_id] = txn
        self._undo.push(("ADD", txn))
        self._next_id += 1

        self.save()
        return txn

    def delete_transaction(self, txn_id: int) -> Transaction:
        """Delete an existing transaction and reverse its balance impact.

        Raises:
            TransactionNotFoundError: If txn_id is not in index.
        """
        if txn_id not in self._by_id:
            raise TransactionNotFoundError(f"Transaction #{txn_id} does not exist.")

        txn = self._by_id[txn_id]
        acc = self.get_or_create_account(txn.account)

        # Reversing income may require funds check; Account.reverse handles this
        acc.reverse(txn)

        # Remove from index and list
        del self._by_id[txn_id]
        self._transactions.remove(txn)

        self._undo.push(("DELETE", txn))
        self.save()
        return txn

    def update_transaction(self, txn_id: int, **updates: Any) -> tuple[Transaction, Transaction]:
        """Update fields of an existing transaction and adjust balances accordingly.

        Returns:
            Tuple of (old_transaction, new_transaction).

        Raises:
            TransactionNotFoundError: If txn_id does not exist.
            InsufficientFundsError: If updated amount causes balance deficiency.
        """
        if txn_id not in self._by_id:
            raise TransactionNotFoundError(f"Transaction #{txn_id} does not exist.")

        old_txn = self._by_id[txn_id]
        data = old_txn.to_dict()
        data.update(updates)
        data["txn_id"] = txn_id  # ensure ID cannot be mutated

        new_txn = Transaction.from_dict(data)

        # Handle account balance adjustments atomically
        old_acc = self.get_or_create_account(old_txn.account)
        new_acc = self.get_or_create_account(new_txn.account)

        # Reverse old transaction
        old_acc.reverse(old_txn)
        try:
            # Apply new transaction
            new_acc.apply(new_txn)
        except InsufficientFundsError:
            # Rollback old transaction application if new one fails
            old_acc.apply(old_txn)
            raise

        # Update collections
        self._transactions.remove(old_txn)
        bisect.insort(self._transactions, new_txn)
        self._by_id[txn_id] = new_txn

        self._undo.push(("UPDATE", old_txn, new_txn))
        self.save()
        return old_txn, new_txn

    # -------------------------------------------------------------------------
    # Undo System
    # -------------------------------------------------------------------------
    def undo(self) -> str:
        """Revert the most recent ADD, DELETE, or UPDATE operation (LIFO).

        Returns:
            Human-readable message describing the undone action.
        """
        if self._undo.is_empty():
            return "Nothing to undo."

        record = self._undo.pop()
        action_type = record[0]

        if action_type == "ADD":
            txn: Transaction = record[1]
            acc = self.get_or_create_account(txn.account)
            acc.reverse(txn)
            if txn in self._transactions:
                self._transactions.remove(txn)
            self._by_id.pop(txn.txn_id, None)
            self.save()
            return f"Undid ADD: Removed transaction #{txn.txn_id:04d} ({txn.category}, ₹{txn.amount:,.2f})"

        elif action_type == "DELETE":
            txn = record[1]
            acc = self.get_or_create_account(txn.account)
            acc.apply(txn)
            bisect.insort(self._transactions, txn)
            self._by_id[txn.txn_id] = txn
            self.save()
            return f"Undid DELETE: Restored transaction #{txn.txn_id:04d} ({txn.category}, ₹{txn.amount:,.2f})"

        elif action_type == "UPDATE":
            old_txn: Transaction = record[1]
            new_txn: Transaction = record[2]

            old_acc = self.get_or_create_account(old_txn.account)
            new_acc = self.get_or_create_account(new_txn.account)

            new_acc.reverse(new_txn)
            old_acc.apply(old_txn)

            if new_txn in self._transactions:
                self._transactions.remove(new_txn)
            bisect.insort(self._transactions, old_txn)
            self._by_id[old_txn.txn_id] = old_txn

            self.save()
            return f"Undid UPDATE: Reverted transaction #{old_txn.txn_id:04d} back to previous state"

        return "Unknown action type in undo stack."

    # -------------------------------------------------------------------------
    # Query & Search Operations
    # -------------------------------------------------------------------------
    def get_transaction(self, txn_id: int) -> Transaction:
        """O(1) hash map lookup for a transaction by its ID."""
        if txn_id not in self._by_id:
            raise TransactionNotFoundError(f"Transaction #{txn_id} not found.")
        return self._by_id[txn_id]

    def get_all_transactions(self) -> list[Transaction]:
        """Return all transactions sorted chronologically."""
        return list(self._transactions)

    def search_transactions(self, query: str) -> list[Transaction]:
        """Search transactions by text query matching description, category, or account."""
        clean_q = (query or "").strip().lower()
        if not clean_q:
            return []

        results = []
        for txn in self._transactions:
            if (
                clean_q in txn.description.lower()
                or clean_q in txn.category.lower()
                or clean_q in txn.account.lower()
                or clean_q in txn.txn_type.lower()
                or clean_q == str(txn.amount)
            ):
                results.append(txn)
        return results

    def filter_by_date(self, start_date: str, end_date: str) -> list[Transaction]:
        """O(log n) date range query using binary search."""
        return transactions_in_range(self._transactions, start_date, end_date)

    def get_top_expenses(self, n: int = 5) -> list[Transaction]:
        """O(m log n) Top-N expenses using Min-Heap."""
        return top_n_expenses(self._transactions, n=n)

    def get_account_balances(self) -> dict[str, float]:
        """Return current balances for all accounts."""
        return {name: acc.balance for name, acc in self.accounts.items()}

    def describe_accounts(self) -> list[str]:
        """Return polymorphic descriptions for all managed accounts."""
        return [acc.describe() for acc in self.accounts.values()]
