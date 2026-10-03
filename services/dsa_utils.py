"""Data Structures and Algorithms utility module for Smart Expense Tracker.

Explicitly implements:
1. Hash Map (dict) for O(n) category total aggregations and O(1) ID lookups.
2. Min-Heap (heapq) for O(m log n) Top-N largest expenses retrieval without full sorting.
3. Binary Search (bisect) for O(log n) date boundary range queries on sorted transactions.
4. Stack (LIFO) for action history tracking and undo functionality.
"""

from __future__ import annotations

import bisect
import heapq
from typing import Any, Sequence

from models.transaction import Transaction


# -----------------------------------------------------------------------------
# 1. HASH MAP: Category Aggregation & Lookup (Time: O(n), Space: O(k))
# -----------------------------------------------------------------------------
def category_totals(
    transactions: Sequence[Transaction], txn_type: str = "expense"
) -> dict[str, float]:
    """Aggregate total transaction amounts by category using a hash map (dict).

    Time Complexity: O(n), where n is the number of transactions.
    Space Complexity: O(k), where k is the number of unique categories.

    Args:
        transactions: Sequence of transactions.
        txn_type: Type of transactions to aggregate ('expense' or 'income').

    Returns:
        Dictionary mapping category names to sum of amounts (rounded to 2 decimals).
    """
    totals: dict[str, float] = {}
    target_type = txn_type.strip().lower()

    for txn in transactions:
        if txn.txn_type == target_type:
            cat = txn.category
            totals[cat] = round(totals.get(cat, 0.0) + txn.amount, 2)

    return totals


def build_id_index(transactions: Sequence[Transaction]) -> dict[int, Transaction]:
    """Build a hash map index mapping transaction ID to Transaction object.

    Enables O(1) direct lookup by ID.
    """
    return {txn.txn_id: txn for txn in transactions}


# -----------------------------------------------------------------------------
# 2. HEAP: Top-N Expenses (Time: O(m log n), Space: O(n))
# -----------------------------------------------------------------------------
def top_n_expenses(transactions: Sequence[Transaction], n: int = 5) -> list[Transaction]:
    """Retrieve the top N largest expense transactions using a Min-Heap.

    ALGORITHM EXPLANATION:
    Instead of sorting all m expenses (which takes O(m log m) time and O(m) space),
    we maintain a Min-Heap of size at most N:
    1. Filter for expense transactions (ignore income).
    2. For each expense, compare with the minimum element at heap root (heap[0]).
    3. If the current amount is larger than the root, replace the root using heapreplace (O(log n)).
    4. At the end, the heap contains the top N largest items.
    5. Sort the N items descending for final presentation (O(n log n)).

    Time Complexity: O(m log n) where m is the count of expenses, n is the top count.
    Space Complexity: O(n) for the heap elements.

    Args:
        transactions: Sequence of transactions.
        n: Number of top expenses to return.

    Returns:
        List of top N expense transactions sorted descending by amount.
    """
    if n <= 0:
        return []

    # Min-heap storing tuples: (amount, txn_id, txn)
    # Including txn_id breaks ties cleanly without comparing Transaction objects directly.
    min_heap: list[tuple[float, int, Transaction]] = []

    for txn in transactions:
        if txn.txn_type != "expense":
            continue

        item = (txn.amount, txn.txn_id, txn)
        if len(min_heap) < n:
            heapq.heappush(min_heap, item)
        elif txn.amount > min_heap[0][0]:
            heapq.heapreplace(min_heap, item)

    # Extract items and sort descending by amount
    result = [entry[2] for entry in min_heap]
    result.sort(key=lambda t: t.amount, reverse=True)
    return result


# -----------------------------------------------------------------------------
# 3. BINARY SEARCH: Date Range Queries (Time: O(log n + k), Space: O(k))
# -----------------------------------------------------------------------------
def transactions_in_range(
    sorted_transactions: Sequence[Transaction],
    start_date: str,
    end_date: str,
) -> list[Transaction]:
    """Find transactions within [start_date, end_date] inclusive via binary search.

    PRECONDITION:
    `sorted_transactions` must already be ordered chronologically by `(txn_date, txn_id)`.

    ALGORITHM EXPLANATION:
    Uses `bisect.bisect_left` to find the first transaction with date >= start_date in O(log n),
    and `bisect.bisect_right` to find the insertion point for end_date in O(log n).
    The slice between these indices is the matching range.

    Time Complexity: O(log n + k) where n is total transactions, k is matching count.
    Space Complexity: O(k) for the result slice.

    Args:
        sorted_transactions: Chronologically sorted transaction list.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).

    Returns:
        List of matching Transaction instances.
    """
    if not sorted_transactions or start_date > end_date:
        return []

    # Find left boundary: first element with txn_date >= start_date
    left_idx = bisect.bisect_left(
        sorted_transactions,
        start_date,
        key=lambda t: t.txn_date,
    )

    # Find right boundary: first element with txn_date > end_date
    right_idx = bisect.bisect_right(
        sorted_transactions,
        end_date,
        key=lambda t: t.txn_date,
    )

    return list(sorted_transactions[left_idx:right_idx])


# -----------------------------------------------------------------------------
# 4. STACK: LIFO Undo Structure (Time: O(1) push/pop, Space: O(u))
# -----------------------------------------------------------------------------
class UndoStack:
    """Last-In-First-Out (LIFO) Stack implementation for tracking and undoing changes.

    Supports push, pop, peek, is_empty, len, and clear in O(1) time complexity.
    """

    def __init__(self) -> None:
        self._items: list[Any] = []

    def push(self, item: Any) -> None:
        """Push an action or state record onto the stack."""
        self._items.append(item)

    def pop(self) -> Any:
        """Remove and return the most recently pushed item.

        Raises:
            IndexError: If the stack is empty.
        """
        if self.is_empty():
            raise IndexError("pop from empty UndoStack")
        return self._items.pop()

    def peek(self) -> Any | None:
        """Return top element without removing it, or None if empty."""
        return self._items[-1] if not self.is_empty() else None

    def is_empty(self) -> bool:
        """Check whether the stack contains any items."""
        return len(self._items) == 0

    def clear(self) -> None:
        """Empty the stack."""
        self._items.clear()

    def __len__(self) -> int:
        """Return the number of items on the stack."""
        return len(self._items)

    def __repr__(self) -> str:
        return f"UndoStack(size={len(self._items)})"
