"""CSV file persistence service for Smart Expense Tracker.

Uses Python's built-in csv module to read and write transaction records.
Includes robust integrity checks to detect and report corrupted files.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from models.exceptions import ExpenseTrackerError, StorageError
from models.transaction import Transaction

CSV_FIELDNAMES = [
    "txn_id",
    "txn_date",
    "amount",
    "category",
    "description",
    "txn_type",
    "account",
]


def load_transactions(file_path: Path | str) -> list[Transaction]:
    """Load transactions from a CSV file.

    Args:
        file_path: Path to CSV storage file.

    Returns:
        List of Transaction instances. Returns empty list if file does not exist.

    Raises:
        StorageError: If CSV is unreadable, corrupted, or has invalid data.
    """
    path = Path(file_path)
    if not path.is_file():
        return []

    transactions: list[Transaction] = []
    try:
        with open(path, mode="r", encoding="utf-8", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                return []

            # Check that required columns are in fieldnames
            missing = [f for f in CSV_FIELDNAMES if f not in reader.fieldnames]
            if missing:
                raise StorageError(f"Corrupted CSV header in '{path}': missing columns {missing}")

            for line_no, row in enumerate(reader, start=2):
                if not any(row.values()):
                    continue  # skip completely blank rows
                try:
                    txn = Transaction.from_dict(row)
                    transactions.append(txn)
                except ExpenseTrackerError as err:
                    raise StorageError(
                        f"Corrupted CSV row #{line_no} in '{path}': {err}"
                    ) from err
                except Exception as err:
                    raise StorageError(
                        f"Unexpected error parsing row #{line_no} in '{path}': {err}"
                    ) from err

    except OSError as err:
        raise StorageError(f"Failed to read CSV file '{path}': {err}") from err

    return transactions


def save_transactions(transactions: Sequence[Transaction], file_path: Path | str) -> None:
    """Save transactions to a CSV file.

    Args:
        transactions: Sequence of Transaction instances.
        file_path: Destination path for CSV file.

    Raises:
        StorageError: If directory creation or file writing fails.
    """
    path = Path(file_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            for txn in transactions:
                writer.writerow(txn.to_dict())
    except OSError as err:
        raise StorageError(f"Failed to write CSV file '{path}': {err}") from err
