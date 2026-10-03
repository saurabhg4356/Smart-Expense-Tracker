"""Custom exceptions for Smart Expense Tracker."""


class ExpenseTrackerError(Exception):
    """Base exception for all expense tracker domain errors."""
    pass


class InvalidAmountError(ExpenseTrackerError):
    """Raised when a monetary amount is non-numeric, non-finite, or <= 0."""
    pass


class InvalidTransactionError(ExpenseTrackerError):
    """Raised when transaction data (type, date, account, etc.) fails validation."""
    pass


class InsufficientFundsError(ExpenseTrackerError):
    """Raised when an account withdrawal exceeds available balance or falls below minimum balance."""
    pass


class TransactionNotFoundError(ExpenseTrackerError):
    """Raised when a transaction with the requested ID does not exist."""
    pass


class StorageError(ExpenseTrackerError):
    """Raised when CSV reading, writing, or parsing encounters an I/O or integrity failure."""
    pass


class NotEnoughDataError(ExpenseTrackerError):
    """Raised when there is insufficient historical data for analysis or ML modeling."""
    pass
