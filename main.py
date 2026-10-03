"""Command-Line Interface (CLI) for Smart Expense Tracker.

Entry point supporting both interactive operation and --demo bootstrap mode.
"""

from __future__ import annotations

import argparse
import datetime
import shutil
import sys
from pathlib import Path

# Reconfigure standard output streams to UTF-8 on Windows to safely display
# currency symbols (₹) and Unicode indicators without cp1252 UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from models.exceptions import (
    ExpenseTrackerError,
    InsufficientFundsError,
    InvalidAmountError,
    InvalidTransactionError,
    NotEnoughDataError,
    StorageError,
    TransactionNotFoundError,
)
from models.transaction import Transaction
from services.analytics import detect_anomalies, summary_stats
from services.expense_manager import ExpenseManager
from services.predictor import SpendingPredictor
from services.visualizer import plot_category_pie, plot_monthly_trend

SAMPLE_DATA_PATH = Path("data/sample_data.csv")
RUNTIME_DATA_PATH = Path("data/transactions.csv")



def prompt_string(prompt_text: str, default: str = "") -> str:
    """Read a stripped string with an optional default."""
    display = f"{prompt_text} [{default}]: " if default else f"{prompt_text}: "
    val = input(display).strip()
    return val if val else default


def prompt_float(prompt_text: str) -> float:
    """Read and validate a positive floating-point number."""
    while True:
        raw = input(f"{prompt_text}: ").strip()
        try:
            val = float(raw)
            if val <= 0:
                print(" [!] Amount must be strictly greater than 0. Try again.")
                continue
            return round(val, 2)
        except ValueError:
            print(" [!] Invalid numeric value. Please enter a valid number.")


def prompt_choice(prompt_text: str, choices: list[str], default: str) -> str:
    """Read a string that matches one of the specified choices."""
    choices_lower = [c.lower() for c in choices]
    while True:
        display = f"{prompt_text} ({'/'.join(choices)}) [{default}]: "
        raw = input(display).strip().lower()
        if not raw:
            return default.lower()
        if raw in choices_lower:
            return raw
        print(f" [!] Invalid option '{raw}'. Choose from {choices}.")


def display_transactions_table(transactions: list[Transaction], title: str = "TRANSACTIONS") -> None:
    """Format and print a tabular view of transactions."""
    if not transactions:
        print("\n [i] No transactions found.")
        return

    print(f"\n--- {title} ({len(transactions)} records) ---")
    header = (
        f"{'ID':<6} {'Date':<12} {'Type':<8} {'Amount':>12}  "
        f"{'Category':<15} {'Account':<10} {'Description'}"
    )
    print(header)
    print("-" * len(header))

    for txn in transactions:
        sign = "+" if txn.txn_type == "income" else "-"
        amt_str = f"{sign}₹{txn.amount:,.2f}"
        print(
            f"{txn.txn_id:<6} {txn.txn_date:<12} {txn.txn_type:<8} {amt_str:>12}  "
            f"{txn.category:<15} {txn.account:<10} {txn.description}"
        )
    print("-" * len(header))


# -----------------------------------------------------------------------------
# Individual Menu Action Handlers
# -----------------------------------------------------------------------------
def action_add(manager: ExpenseManager) -> None:
    """Action 1: Add a new transaction."""
    print("\n--- Add Transaction ---")
    today = datetime.date.today().isoformat()
    date_str = prompt_string("Transaction Date (YYYY-MM-DD)", default=today)
    amount = prompt_float("Amount (₹)")
    txn_type = prompt_choice("Transaction Type", ["expense", "income"], default="expense")
    account = prompt_string("Account Name (e.g. Wallet, Bank, Savings)", default="Wallet")
    description = prompt_string("Description / Memo", default="")

    # Suggest category via ML if user leaves category blank
    predicted_cat, conf = manager.categorizer.predict_with_confidence(description)
    cat_default = predicted_cat if predicted_cat != "Other" else "Other"

    category_input = prompt_string(
        f"Category (leave blank to use auto-detected '{cat_default}')",
        default="",
    )
    category = category_input if category_input else cat_default

    try:
        txn = manager.add_transaction(
            txn_date=date_str,
            amount=amount,
            category=category,
            description=description,
            txn_type=txn_type,
            account=account,
        )
        print(f"\n [✓] Successfully recorded transaction #{txn.txn_id:04d}!")
        print(f"     {txn}")
    except (InsufficientFundsError, InvalidAmountError, InvalidTransactionError) as err:
        print(f"\n [✗] Failed to add transaction: {err}")


def action_list(manager: ExpenseManager) -> None:
    """Action 2: List all transactions."""
    txns = manager.get_all_transactions()
    display_transactions_table(txns, "ALL TRANSACTIONS")


def action_delete(manager: ExpenseManager) -> None:
    """Action 3: Delete a transaction by ID."""
    print("\n--- Delete Transaction ---")
    raw_id = prompt_string("Enter Transaction ID to delete")
    try:
        txn_id = int(raw_id)
        deleted = manager.delete_transaction(txn_id)
        print(f"\n [✓] Deleted transaction #{deleted.txn_id:04d}:")
        print(f"     {deleted}")
        print(f"     (Balance for account '{deleted.account}' has been reversed)")
    except ValueError:
        print(" [!] Transaction ID must be an integer.")
    except (TransactionNotFoundError, InsufficientFundsError) as err:
        print(f"\n [✗] {err}")


def action_edit(manager: ExpenseManager) -> None:
    """Action 4: Edit an existing transaction."""
    print("\n--- Edit Transaction ---")
    raw_id = prompt_string("Enter Transaction ID to edit")
    try:
        txn_id = int(raw_id)
        old_txn = manager.get_transaction(txn_id)
    except ValueError:
        print(" [!] Transaction ID must be an integer.")
        return
    except TransactionNotFoundError as err:
        print(f"\n [✗] {err}")
        return

    print(f"\nCurrent record: {old_txn}")
    print("Press [Enter] on any field to keep current value.\n")

    new_date = prompt_string("New Date (YYYY-MM-DD)", default=old_txn.txn_date)
    raw_amt = prompt_string("New Amount (₹)", default=str(old_txn.amount))
    new_type = prompt_choice("New Type", ["expense", "income"], default=old_txn.txn_type)
    new_cat = prompt_string("New Category", default=old_txn.category)
    new_acc = prompt_string("New Account", default=old_txn.account)
    new_desc = prompt_string("New Description", default=old_txn.description)

    try:
        amt_float = float(raw_amt)
        _, updated = manager.update_transaction(
            txn_id=txn_id,
            txn_date=new_date,
            amount=amt_float,
            txn_type=new_type,
            category=new_cat,
            account=new_acc,
            description=new_desc,
        )
        print(f"\n [✓] Transaction #{txn_id:04d} updated successfully:")
        print(f"     {updated}")
    except (ExpenseTrackerError, ValueError) as err:
        print(f"\n [✗] Failed to update transaction: {err}")


def action_search(manager: ExpenseManager) -> None:
    """Action 5: Search transactions by keyword."""
    print("\n--- Search Transactions ---")
    query = prompt_string("Enter search term (description, category, or account)")
    if not query:
        print(" [!] Empty query entered.")
        return
    results = manager.search_transactions(query)
    display_transactions_table(results, f"SEARCH RESULTS FOR '{query}'")


def action_undo(manager: ExpenseManager) -> None:
    """Action 6: Undo last state change."""
    print("\n--- Undo Last Action ---")
    msg = manager.undo()
    print(f" [i] {msg}")


def action_balances(manager: ExpenseManager) -> None:
    """Action 7: Display account balances."""
    print("\n--- Account Balances ---")
    descriptions = manager.describe_accounts()
    for desc in descriptions:
        print(f" • {desc}")


def action_analytics(manager: ExpenseManager) -> None:
    """Action 8: Display comprehensive analytics summary."""
    print("\n--- Analytics Summary ---")
    txns = manager.get_all_transactions()
    stats = summary_stats(txns)

    print(f" Total Transactions   : {stats['transaction_count']}")
    print(f" Total Income         : ₹{stats['total_income']:,.2f}")
    print(f" Total Expenses       : ₹{stats['total_expenses']:,.2f}")
    print(f" Net Savings          : ₹{stats['net_savings']:,.2f}")
    print(f" Savings Rate         : {stats['savings_rate']:.1f}%")
    print(f" Average Expense      : ₹{stats['average_expense']:,.2f}")
    print(f" Median Expense       : ₹{stats['median_expense']:,.2f}")
    print(f" Std Dev (Expense)    : ₹{stats['std_expense']:,.2f}")
    print(f" Largest Expense      : ₹{stats['largest_expense']:,.2f}")
    print(f" Highest Spending Day : {stats['highest_spending_day']}")

    if stats["category_breakdown"]:
        print("\n Category Spending Breakdown:")
        for cat, amt in stats["category_breakdown"].items():
            print(f"   - {cat:<15} : ₹{amt:>10,.2f}")

    if stats["monthly_spending"]:
        print("\n Monthly Expenses Trend:")
        for m, amt in stats["monthly_spending"].items():
            print(f"   - {m:<10} : ₹{amt:>10,.2f}")


def action_anomalies(manager: ExpenseManager) -> None:
    """Action 9: Detect unusual expenses using category z-score."""
    print("\n--- Unusual Expense Detection (Z-Score > 2.0) ---")
    txns = manager.get_all_transactions()
    anomalies = detect_anomalies(txns, threshold=2.0)

    if not anomalies:
        print(" [✓] No anomalous transactions detected (all spending within normal z-score range).")
        return

    print(f" [!] Detected {len(anomalies)} statistical outlier(s):\n")
    for item in anomalies:
        print(
            f" • ID #{item['txn_id']:04d} | {item['date']} | ₹{item['amount']:,.2f} | "
            f"Cat: {item['category']} | Z-Score: {item['z_score']:+.2f}\n"
            f"   Desc: {item['description']}\n"
            f"   (Category Mean: ₹{item['category_mean']:,.2f}, Std: ₹{item['category_std']:,.2f})"
        )


def action_top_expenses(manager: ExpenseManager) -> None:
    """Action 10: Top expenses using Min-Heap."""
    print("\n--- Top Expenses (Min-Heap Priority Queue) ---")
    raw_n = prompt_string("Number of top expenses to display", default="5")
    try:
        n = int(raw_n)
    except ValueError:
        n = 5
    top_items = manager.get_top_expenses(n=n)
    display_transactions_table(top_items, f"TOP {n} LARGEST EXPENSES")


def action_date_range(manager: ExpenseManager) -> None:
    """Action 11: Date range query using Binary Search."""
    print("\n--- Search by Date Range (Binary Search) ---")
    start = prompt_string("Start Date (YYYY-MM-DD)")
    end = prompt_string("End Date (YYYY-MM-DD)")
    if start > end:
        print(" [!] Start date cannot be after end date.")
        return
    matches = manager.filter_by_date(start, end)
    display_transactions_table(matches, f"TRANSACTIONS BETWEEN {start} AND {end}")


def action_predict(manager: ExpenseManager) -> None:
    """Action 12: Predict next month's spending with Linear Regression."""
    print("\n--- Spending Prediction (Linear Regression) ---")
    txns = manager.get_all_transactions()
    stats = summary_stats(txns)
    monthly_data = stats.get("monthly_spending", {})

    if len(monthly_data) < SpendingPredictor.MIN_MONTHS_REQUIRED:
        print(
            f" [!] Insufficient data: Found {len(monthly_data)} month(s) of expense history.\n"
            f"     At least {SpendingPredictor.MIN_MONTHS_REQUIRED} distinct months of expense records "
            f"are required to fit the regression model."
        )
        return

    predictor = SpendingPredictor()
    try:
        predictor.fit(monthly_data)
        predicted_amt, mae, next_month = predictor.predict_next()
        print(f"\n [★] Model Forecast for Next Period ({next_month}):")
        print(f"     Estimated Spending     : ₹{predicted_amt:,.2f}")
        print(f"     In-Sample MAE          : ±₹{mae:,.2f}")
        print("\n [i] NOTE ON MODEL INTERPRETATION:")
        print("     The Mean Absolute Error (MAE) reported is calculated strictly in-sample on the")
        print("     available training months. In educational settings, in-sample error is typically")
        print("     optimistic and serves to demonstrate the regression fit rather than guarantee")
        print("     financial precision against real-world volatility.")
    except NotEnoughDataError as err:
        print(f" [!] Prediction failed: {err}")


def action_charts(manager: ExpenseManager) -> None:
    """Action 13: Generate Matplotlib visual charts."""
    print("\n--- Generate Visual Charts ---")
    txns = manager.get_all_transactions()
    if not txns:
        print(" [!] No transactions available to plot.")
        return

    # Check if prediction is available to overlay on trend
    predicted_tuple: tuple[float, str] | None = None
    stats = summary_stats(txns)
    monthly_data = stats.get("monthly_spending", {})
    if len(monthly_data) >= SpendingPredictor.MIN_MONTHS_REQUIRED:
        try:
            pred = SpendingPredictor().fit(monthly_data)
            pred_amt, _, next_m = pred.predict_next()
            predicted_tuple = (pred_amt, next_m)
        except Exception:
            predicted_tuple = None

    pie_file = plot_category_pie(txns, "charts/category_pie.png")
    trend_file = plot_monthly_trend(txns, predicted_next=predicted_tuple, output_path="charts/monthly_trend.png")

    if pie_file:
        print(f" [✓] Category pie chart generated : {pie_file}")
    else:
        print(" [!] Unable to generate category pie chart (no expense data).")

    if trend_file:
        print(f" [✓] Monthly trend chart generated: {trend_file}")
        if predicted_tuple:
            print(f"     (Included ML projection for {predicted_tuple[1]}: ₹{predicted_tuple[0]:,.0f})")
    else:
        print(" [!] Unable to generate monthly trend chart (no expense data).")


def action_test_categorizer(manager: ExpenseManager) -> None:
    """Action 14: Test automated categorization with confidence scores."""
    print("\n--- Automatic Category Classifier (TF-IDF + Naive Bayes) ---")
    test_text = prompt_string("Enter sample transaction description to classify")
    if not test_text:
        print(" [!] No description entered.")
        return

    cat, conf = manager.categorizer.predict_with_confidence(test_text)
    print(f"\n [★] Classification Result for '{test_text}':")
    print(f"     Assigned Category : {cat}")
    print(f"     Confidence Score  : {conf * 100:.1f}%")
    if cat == "Other" and conf < manager.categorizer.CONFIDENCE_THRESHOLD:
        print(f"     (Fell below threshold of {manager.categorizer.CONFIDENCE_THRESHOLD * 100:.0f}%, defaulted to 'Other')")


# -----------------------------------------------------------------------------
# Main Application Loop & Entrypoint
# -----------------------------------------------------------------------------
MENU_TEXT = """
========== SMART EXPENSE TRACKER ==========
 1. Add transaction
 2. List transactions
 3. Delete transaction
 4. Edit transaction
 5. Search
 6. Undo last change
 7. Account balances
 8. Analytics summary
 9. Detect unusual expenses
10. Top expenses
11. Transactions in a date range
12. Predict next month's spending
13. Show charts
14. Test auto-categorizer
 0. Exit
===========================================
"""


def bootstrap_demo_data() -> None:
    """Bootstrap demo dataset into runtime storage if not already present."""
    if not RUNTIME_DATA_PATH.exists() and SAMPLE_DATA_PATH.exists():
        RUNTIME_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SAMPLE_DATA_PATH, RUNTIME_DATA_PATH)
        print(f"[i] Demo mode: Initialized '{RUNTIME_DATA_PATH}' from '{SAMPLE_DATA_PATH}'.")
    elif RUNTIME_DATA_PATH.exists():
        print(f"[i] Demo mode: Using existing runtime file '{RUNTIME_DATA_PATH}'.")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Smart Expense Tracker — OOP + DSA + ML CLI")
    parser.add_argument("--demo", action="store_true", help="Bootstrap with sample data if needed")
    args = parser.parse_args()

    if args.demo:
        bootstrap_demo_data()

    manager = ExpenseManager(storage_path=RUNTIME_DATA_PATH)

    actions = {
        "1": action_add,
        "2": action_list,
        "3": action_delete,
        "4": action_edit,
        "5": action_search,
        "6": action_undo,
        "7": action_balances,
        "8": action_analytics,
        "9": action_anomalies,
        "10": action_top_expenses,
        "11": action_date_range,
        "12": action_predict,
        "13": action_charts,
        "14": action_test_categorizer,
    }

    print("Welcome to Smart Expense Tracker!")
    if args.demo:
        print("Demo dataset loaded with 6 months of sample financial records.")

    while True:
        try:
            print(MENU_TEXT)
            choice = input("Enter choice (0-14): ").strip()
            if choice == "0":
                print("\nThank you for using Smart Expense Tracker. Goodbye!")
                break
            action_fn = actions.get(choice)
            if action_fn:
                action_fn(manager)
            else:
                print(f" [!] Invalid choice '{choice}'. Please select a number from 0 to 14.")
        except (KeyboardInterrupt, EOFError):
            print("\n\nOperation aborted by user. Exiting...")
            break
        except Exception as err:
            print(f"\n [✗] An unexpected error occurred: {err}")


if __name__ == "__main__":
    main()
