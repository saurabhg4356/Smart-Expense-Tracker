"""Machine Learning Feature #2: Transaction Categorizer using TF-IDF and Naive Bayes.

Combines TF-IDF feature extraction with Multinomial Naive Bayes inside a scikit-learn Pipeline.
Pre-seeded with diverse real-world descriptions and capable of incremental retraining
from user transaction history.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from models.transaction import Transaction

# Built-in seed training examples covering major spending categories
DEFAULT_TRAINING_CORPUS: list[tuple[str, str]] = [
    # Food
    ("swiggy food order biryani", "Food"),
    ("zomato dinner delivery pasta", "Food"),
    ("mcdonalds burger meal fries", "Food"),
    ("starbucks coffee latte cappuccino", "Food"),
    ("dominos pizza dinner coke", "Food"),
    ("subway sandwich lunch meal", "Food"),
    ("restaurant dining dinner family", "Food"),
    ("cafe coffee day beverage snacks", "Food"),
    ("local street food chai snacks", "Food"),
    ("bakery pastry cakes bread", "Food"),

    # Groceries
    ("dmart supermarket provisions store", "Groceries"),
    ("blinkit quick grocery vegetables fruits", "Groceries"),
    ("zepto milk butter bread eggs", "Groceries"),
    ("bigbasket monthly grocery essentials", "Groceries"),
    ("reliance fresh organic vegetables", "Groceries"),
    ("kirana store daily staples rice pulses", "Groceries"),
    ("nature basket gourmet olive oil cheese", "Groceries"),
    ("supermarket household grains spices oil", "Groceries"),

    # Transport
    ("uber cab ride airport trip", "Transport"),
    ("ola auto ride daily commute", "Transport"),
    ("metro rail smart card recharge travel", "Transport"),
    ("petrol fuel bunk indian oil hpcl", "Transport"),
    ("shell diesel tank refill petrol", "Transport"),
    ("irctc train ticket reservation railway", "Transport"),
    ("flight ticket airline booking indigo", "Transport"),
    ("fastag highway toll plaza payment", "Transport"),
    ("bus transit card pass commute", "Transport"),

    # Utilities
    ("electricity bill payment bescom power", "Utilities"),
    ("broadband internet wifi bill airtel fiber", "Utilities"),
    ("piped gas cylinder lpg refill bill", "Utilities"),
    ("mobile recharge prepaid jio vi airtel", "Utilities"),
    ("water supply utility municipality cess", "Utilities"),
    ("tataplay dth cable tv subscription", "Utilities"),

    # Housing
    ("monthly house rent owner apartment", "Housing"),
    ("society maintenance charges building", "Housing"),
    ("home loan emi bank mortgage", "Housing"),
    ("flat rental deposit lease advance", "Housing"),
    ("plumbing electrician carpentry repairs home", "Housing"),
    ("house painting renovation maintenance", "Housing"),

    # Entertainment
    ("netflix monthly streaming subscription plan", "Entertainment"),
    ("bookmyshow movie cinema multiplex ticket", "Entertainment"),
    ("spotify premium music subscription", "Entertainment"),
    ("amazon prime video subscription annual", "Entertainment"),
    ("steam playstation gaming console games", "Entertainment"),
    ("concert live music show ticket", "Entertainment"),
    ("board games sports club weekend outing", "Entertainment"),

    # Shopping
    ("amazon online shopping electronics clothes", "Shopping"),
    ("flipkart apparel footwear fashion sale", "Shopping"),
    ("myntra jackets jeans sneakers order", "Shopping"),
    ("zara casual cotton shirt trousers", "Shopping"),
    ("uniqlo winter jacket fleece hoodie", "Shopping"),
    ("apple store iphone macbook accessories", "Shopping"),
    ("ikea furniture home decor utensils", "Shopping"),

    # Health
    ("apollo pharmacy medicine tablets syrup", "Health"),
    ("doctor clinic consultation fees physician", "Health"),
    ("medplus wellness prescription drugs", "Health"),
    ("blood test laboratory pathology health check", "Health"),
    ("dental checkup tooth extraction dentist", "Health"),
    ("gym membership fitness crossfit center", "Health"),
    ("health insurance mediclaim annual premium", "Health"),

    # Salary
    ("monthly salary tech solutions corp", "Salary"),
    ("company payroll direct deposit salary", "Salary"),
    ("quarterly performance bonus incentive", "Salary"),
    ("freelance client consulting payout wire", "Salary"),
    ("internship monthly stipend allowance", "Salary"),
]


class TransactionCategorizer:
    """Classifies transaction descriptions into financial categories using TF-IDF + Naive Bayes.

    Enforces a minimum confidence threshold (default 0.35). Predictions falling below
    the threshold default to 'Other'.
    """

    CONFIDENCE_THRESHOLD = 0.35

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        self._confidence_threshold = float(confidence_threshold)
        self._pipeline: Pipeline | None = None
        self._categories: list[str] = []
        # Auto-train on built-in seed corpus on instantiation
        self._initialize_seed_model()

    def _initialize_seed_model(self) -> None:
        texts = [desc for desc, cat in DEFAULT_TRAINING_CORPUS]
        labels = [cat for desc, cat in DEFAULT_TRAINING_CORPUS]
        self.fit(texts, labels)

    @property
    def is_fitted(self) -> bool:
        """Whether the model pipeline is trained."""
        return self._pipeline is not None

    def fit(self, descriptions: Sequence[str], categories: Sequence[str]) -> TransactionCategorizer:
        """Fit TF-IDF + MultinomialNB pipeline on text descriptions and categories."""
        if not descriptions or not categories or len(descriptions) != len(categories):
            raise ValueError("Descriptions and categories must be non-empty and of identical length.")

        # Ensure text is lowercase string
        cleaned_texts = [str(d).lower().strip() for d in descriptions]
        cleaned_labels = [str(c).strip() for c in categories]

        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ])

        self._pipeline.fit(cleaned_texts, cleaned_labels)
        # Store class labels
        clf = self._pipeline.named_steps["clf"]
        self._categories = list(clf.classes_)
        return self

    def fit_from_transactions(self, transactions: Sequence[Transaction]) -> TransactionCategorizer:
        """Augment seed knowledge with labeled user transactions and retrain."""
        texts = [desc for desc, cat in DEFAULT_TRAINING_CORPUS]
        labels = [cat for desc, cat in DEFAULT_TRAINING_CORPUS]

        for txn in transactions:
            if txn.description and txn.category and txn.category != "Other":
                texts.append(txn.description)
                labels.append(txn.category)

        self.fit(texts, labels)
        return self

    def predict_with_confidence(self, description: str) -> tuple[str, float]:
        """Predict the category and return confidence probability.

        Returns:
            Tuple of (category, confidence).
            If confidence < threshold, category defaults to 'Other'.
        """
        if self._pipeline is None:
            self._initialize_seed_model()

        assert self._pipeline is not None
        clean_text = (description or "").strip().lower()
        if not clean_text:
            return "Other", 0.0

        try:
            proba_arr = self._pipeline.predict_proba([clean_text])[0]
            max_idx = int(np.argmax(proba_arr))
            confidence = float(proba_arr[max_idx])
            best_cat = self._categories[max_idx]

            if confidence >= self._confidence_threshold:
                return best_cat, round(confidence, 2)
            return "Other", round(confidence, 2)
        except Exception:
            return "Other", 0.0

    def predict(self, description: str) -> str:
        """Convenience method returning predicted category string."""
        category, _ = self.predict_with_confidence(description)
        return category
