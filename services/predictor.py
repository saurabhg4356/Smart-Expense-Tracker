"""Machine Learning Feature #1: Monthly Spending Predictor using Linear Regression.

Uses scikit-learn's LinearRegression to model spending trajectory across consecutive months.
Requires at least 3 months of historical expense data to train.
"""

from __future__ import annotations

import datetime
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from models.exceptions import NotEnoughDataError


class SpendingPredictor:
    """Predicts next month's total spending using Linear Regression on monthly indices.

    IMPORTANT EDUCATIONAL NOTE:
    This model serves an educational purpose for undergraduate portfolio demonstration.
    The reported Mean Absolute Error (MAE) is strictly an *in-sample* metric calculated
    on the training data itself. As such, it is optimistic and does not account for
    out-of-sample volatility, unexpected one-time expenditures, or inflation.
    """

    MIN_MONTHS_REQUIRED = 3

    def __init__(self) -> None:
        self._model: LinearRegression | None = None
        self._in_sample_mae: float | None = None
        self._n_months: int = 0
        self._last_month_key: str | None = None
        self._monthly_history: list[tuple[str, float]] = []

    @property
    def in_sample_mae(self) -> float | None:
        """In-sample Mean Absolute Error from model fitting (optimistic metric)."""
        return self._in_sample_mae

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been successfully fitted."""
        return self._model is not None

    def fit(self, monthly_expenses: dict[str, float] | pd.Series | Sequence[float]) -> SpendingPredictor:
        """Fit LinearRegression model on monthly expense sequence.

        Args:
            monthly_expenses: Ordered mapping or sequence of monthly expenses.
                If dict or Series, keys should be 'YYYY-MM' strings sorted chronologically.

        Raises:
            NotEnoughDataError: If fewer than 3 months of data are provided.
        """
        self._monthly_history = []

        if isinstance(monthly_expenses, dict):
            # Sort chronologically by month key
            sorted_items = sorted(monthly_expenses.items(), key=lambda x: str(x[0]))
            values = [float(v) for _, v in sorted_items]
            self._monthly_history = [(str(k), float(v)) for k, v in sorted_items]
        elif isinstance(monthly_expenses, pd.Series):
            sorted_series = monthly_expenses.sort_index()
            values = [float(v) for v in sorted_series.values]
            self._monthly_history = [(str(k), float(v)) for k, v in sorted_series.items()]
        else:
            values = [float(v) for v in monthly_expenses]
            self._monthly_history = [(f"Month-{i+1}", float(v)) for i, v in enumerate(values)]

        if len(values) < self.MIN_MONTHS_REQUIRED:
            raise NotEnoughDataError(
                f"Prediction requires at least {self.MIN_MONTHS_REQUIRED} months of historical data. "
                f"Found only {len(values)} month(s)."
            )

        self._n_months = len(values)
        if self._monthly_history:
            self._last_month_key = self._monthly_history[-1][0]

        # Feature matrix: 0-indexed month sequence [[0], [1], [2], ...]
        X = np.arange(self._n_months, dtype=float).reshape(-1, 1)
        y = np.array(values, dtype=float)

        self._model = LinearRegression()
        self._model.fit(X, y)

        # In-sample MAE evaluation
        fitted_preds = self._model.predict(X)
        self._in_sample_mae = round(float(np.mean(np.abs(y - fitted_preds))), 2)

        return self

    def predict_next(self) -> tuple[float, float, str]:
        """Predict total expenses for the immediately following month.

        Returns:
            Tuple of (predicted_amount, in_sample_mae, next_month_label).

        Raises:
            NotEnoughDataError: If model is not fitted.
        """
        if self._model is None or self._in_sample_mae is None:
            raise NotEnoughDataError("Predictor must be fitted with at least 3 months before prediction.")

        next_idx = np.array([[float(self._n_months)]])
        raw_pred = float(self._model.predict(next_idx)[0])
        predicted_amount = max(0.0, round(raw_pred, 2))

        # Calculate next month label if YYYY-MM format was provided
        next_month_label = f"Month {self._n_months + 1}"
        if self._last_month_key and len(self._last_month_key) == 7 and "-" in self._last_month_key:
            try:
                year_str, month_str = self._last_month_key.split("-")
                year, month = int(year_str), int(month_str)
                if month == 12:
                    next_year = year + 1
                    next_month = 1
                else:
                    next_year = year
                    next_month = month + 1
                next_month_label = f"{next_year:04d}-{next_month:02d}"
            except (ValueError, TypeError):
                pass

        return predicted_amount, self._in_sample_mae, next_month_label
