"""Synthetic bank transaction data generator.

Produces realistic customer, product, and transaction data for
data warehouse development and testing.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIRST_NAMES: List[str] = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Kenneth", "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
    "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
    "Nicholas", "Angela", "Eric", "Shirley", "Jonathan", "Anna", "Stephen", "Brenda",
    "Larry", "Pamela", "Justin", "Emma", "Scott", "Nicole", "Brandon", "Helen",
    "Benjamin", "Samantha", "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet", "Jack", "Catherine",
    "Dennis", "Maria", "Jerry", "Heather", "Tyler", "Diane", "Aaron", "Ruth",
    "Jose", "Julie", "Adam", "Olivia", "Nathan", "Joyce", "Henry", "Virginia",
]

LAST_NAMES: List[str] = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill",
    "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell",
    "Mitchell", "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales",
    "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson",
    "Bailey", "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward",
    "Richardson", "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray",
]

CITIES: List[str] = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
    "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
    "Fort Worth", "Columbus", "Charlotte", "Indianapolis", "San Francisco",
    "Seattle", "Denver", "Washington", "Nashville", "Oklahoma City", "El Paso",
    "Boston", "Portland", "Las Vegas", "Memphis", "Louisville", "Baltimore",
    "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Sacramento", "Mesa",
    "Kansas City", "Atlanta", "Omaha", "Colorado Springs", "Raleigh",
]

STATES: List[str] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

MERCHANT_CATEGORIES: List[Dict[str, float]] = [
    {"category": "Groceries", "weight": 0.20, "avg_amount": 75.0},
    {"category": "Restaurants", "weight": 0.15, "avg_amount": 45.0},
    {"category": "Gas Station", "weight": 0.10, "avg_amount": 50.0},
    {"category": "Online Shopping", "weight": 0.15, "avg_amount": 100.0},
    {"category": "Electronics", "weight": 0.05, "avg_amount": 350.0},
    {"category": "Clothing", "weight": 0.08, "avg_amount": 80.0},
    {"category": "Healthcare", "weight": 0.05, "avg_amount": 200.0},
    {"category": "Utilities", "weight": 0.07, "avg_amount": 150.0},
    {"category": "Travel", "weight": 0.05, "avg_amount": 500.0},
    {"category": "Entertainment", "weight": 0.05, "avg_amount": 60.0},
    {"category": "Education", "weight": 0.03, "avg_amount": 250.0},
    {"category": "Subscriptions", "weight": 0.02, "avg_amount": 15.0},
]

PRODUCT_CATALOG: List[Dict[str, object]] = [
    {"product_id": "CHK001", "product_name": "Basic Checking", "product_type": "checking",
     "interest_rate": 0.01, "monthly_fee": 0.0, "min_balance": 0.0},
    {"product_id": "CHK002", "product_name": "Premium Checking", "product_type": "checking",
     "interest_rate": 0.02, "monthly_fee": 15.0, "min_balance": 1500.0},
    {"product_id": "SAV001", "product_name": "Savings Account", "product_type": "savings",
     "interest_rate": 0.04, "monthly_fee": 0.0, "min_balance": 100.0},
    {"product_id": "SAV002", "product_name": "High-Yield Savings", "product_type": "savings",
     "interest_rate": 0.045, "monthly_fee": 0.0, "min_balance": 1000.0},
    {"product_id": "CC001", "product_name": "Rewards Credit Card", "product_type": "credit_card",
     "interest_rate": 0.18, "monthly_fee": 0.0, "min_balance": 0.0},
    {"product_id": "CC002", "product_name": "Platinum Credit Card", "product_type": "credit_card",
     "interest_rate": 0.15, "monthly_fee": 95.0, "min_balance": 0.0},
    {"product_id": "LN001", "product_name": "Personal Loan", "product_type": "loan",
     "interest_rate": 0.08, "monthly_fee": 0.0, "min_balance": 0.0},
    {"product_id": "LN002", "product_name": "Auto Loan", "product_type": "loan",
     "interest_rate": 0.055, "monthly_fee": 0.0, "min_balance": 0.0},
    {"product_id": "INV001", "product_name": "Money Market", "product_type": "investment",
     "interest_rate": 0.035, "monthly_fee": 10.0, "min_balance": 2500.0},
    {"product_id": "INV002", "product_name": "Certificate of Deposit", "product_type": "investment",
     "interest_rate": 0.05, "monthly_fee": 0.0, "min_balance": 5000.0},
]

TRANSACTION_TYPES: List[str] = [
    "debit", "credit", "transfer", "atm_withdrawal", "direct_deposit",
]


class BankDataGenerator:
    """Generate realistic synthetic bank transaction data.

    Produces customers, products, and transactions with realistic distributions
    and correlations (e.g. higher volumes on weekends, paydays).

    Parameters
    ----------
    n_customers : int
        Number of unique customers to generate.
    n_transactions : int
        Total number of transactions to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_customers: int = 5000, n_transactions: int = 500000,
                 seed: int = 42) -> None:
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info(
            "BankDataGenerator initialised: %d customers, %d transactions, seed=%d",
            n_customers, n_transactions, seed,
        )

    # ------------------------------------------------------------------
    # Customers
    # ------------------------------------------------------------------

    def generate_customers(self) -> pd.DataFrame:
        """Generate customer records with realistic demographics.

        Returns
        -------
        pd.DataFrame
            Columns: customer_id, first_name, last_name, email, phone,
            date_of_birth, address, city, state, zip_code,
            account_open_date, credit_score, annual_income, risk_category.
        """
        logger.info("Generating %d customers …", self.n_customers)

        customer_ids = [f"CUST{str(i).zfill(6)}" for i in range(1, self.n_customers + 1)]
        first_names = self.rng.choice(FIRST_NAMES, size=self.n_customers)
        last_names = self.rng.choice(LAST_NAMES, size=self.n_customers)

        emails = [
            f"{fn.lower()}.{ln.lower()}{self.rng.integers(1, 9999)}@example.com"
            for fn, ln in zip(first_names, last_names)
        ]

        phones = [
            f"({self.rng.integers(200, 999):03d}) {self.rng.integers(200, 999):03d}-{self.rng.integers(1000, 9999):04d}"
            for _ in range(self.n_customers)
        ]

        # Date of birth: ages 18-80
        today = pd.Timestamp.now().normalize()
        min_birth = today - pd.DateOffset(years=80)
        max_birth = today - pd.DateOffset(years=18)
        birth_range_days = (max_birth - min_birth).days
        dobs = [
            min_birth + pd.Timedelta(days=int(d))
            for d in self.rng.integers(0, birth_range_days, size=self.n_customers)
        ]

        cities = self.rng.choice(CITIES, size=self.n_customers)
        states = self.rng.choice(STATES, size=self.n_customers)
        zip_codes = [f"{self.rng.integers(10000, 99999)}" for _ in range(self.n_customers)]
        addresses = [
            f"{self.rng.integers(1, 9999)} {self.rng.choice(['Main', 'Oak', 'Pine', 'Maple', 'Cedar', 'Elm', 'Walnut', 'Spring', 'Hill', 'Lake'])} {self.rng.choice(['St', 'Ave', 'Blvd', 'Dr', 'Ln', 'Ct', 'Rd'])}"
            for _ in range(self.n_customers)
        ]

        # Account opening dates: last 10 years
        open_range_days = 365 * 10
        open_dates = [
            today - pd.Timedelta(days=int(d))
            for d in self.rng.integers(0, open_range_days, size=self.n_customers)
        ]

        # Credit scores: roughly normal, clipped to 300-850
        credit_scores = np.clip(
            self.rng.normal(700, 80, size=self.n_customers), 300, 850
        ).astype(int)

        # Annual income: log-normal
        annual_income = np.clip(
            self.rng.lognormal(mean=10.8, sigma=0.6, size=self.n_customers),
            15_000, 500_000,
        ).round(2)

        # Risk category derived from credit score
        risk_categories = []
        for cs in credit_scores:
            if cs >= 750:
                risk_categories.append("low")
            elif cs >= 650:
                risk_categories.append("medium")
            elif cs >= 550:
                risk_categories.append("high")
            else:
                risk_categories.append("very_high")

        df = pd.DataFrame({
            "customer_id": customer_ids,
            "first_name": first_names,
            "last_name": last_names,
            "email": emails,
            "phone": phones,
            "date_of_birth": pd.to_datetime(dobs),
            "address": addresses,
            "city": cities,
            "state": states,
            "zip_code": zip_codes,
            "account_open_date": pd.to_datetime(open_dates),
            "credit_score": credit_scores,
            "annual_income": annual_income,
            "risk_category": risk_categories,
        })

        logger.info("Generated %d customers.", len(df))
        return df

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------

    def generate_products(self) -> pd.DataFrame:
        """Generate bank product catalog.

        Returns
        -------
        pd.DataFrame
            Columns: product_id, product_name, product_type,
            interest_rate, monthly_fee, min_balance.
        """
        logger.info("Generating product catalog (%d products).", len(PRODUCT_CATALOG))
        df = pd.DataFrame(PRODUCT_CATALOG)
        return df

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def generate_transactions(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate transactions with realistic patterns.

        Patterns include:
        - Higher transaction volumes on weekends
        - Credit card transactions at merchants
        - Savings deposits on paydays (1st and 15th of month)
        - Random transfers between accounts

        Parameters
        ----------
        customers : pd.DataFrame
            Customer data (must contain customer_id).
        products : pd.DataFrame
            Product data (must contain product_id).

        Returns
        -------
        pd.DataFrame
            Columns: transaction_id, customer_id, account_id, transaction_date,
            transaction_type, amount, balance_after, merchant_category,
            description, product_id.
        """
        logger.info("Generating %d transactions …", self.n_transactions)

        rng = self.rng
        n = self.n_transactions

        # --- pick customers and products per transaction ---
        cust_ids = rng.choice(customers["customer_id"].values, size=n)
        prod_ids = rng.choice(products["product_id"].values, size=n)

        # --- transaction dates over last 2 years ---
        end_date = pd.Timestamp.now().normalize()
        start_date = end_date - pd.DateOffset(years=2)
        date_range_days = (end_date - start_date).days
        raw_days = rng.integers(0, date_range_days, size=n)
        txn_dates = pd.to_datetime(start_date) + pd.to_timedelta(raw_days, unit="D")

        # add random times
        random_seconds = rng.integers(0, 86400, size=n)
        txn_dates = txn_dates + pd.to_timedelta(random_seconds, unit="s")

        # --- transaction types ---
        txn_type_weights = [0.40, 0.25, 0.15, 0.10, 0.10]
        txn_types = rng.choice(TRANSACTION_TYPES, size=n, p=txn_type_weights)

        # --- merchant category (mainly for debit / credit) ---
        cat_weights = np.array([m["weight"] for m in MERCHANT_CATEGORIES])
        cat_weights = cat_weights / cat_weights.sum()
        merchant_cats = rng.choice(
            [m["category"] for m in MERCHANT_CATEGORIES], size=n, p=cat_weights,
        )
        avg_amounts = {
            m["category"]: m["avg_amount"] for m in MERCHANT_CATEGORIES
        }

        # --- amounts ---
        amounts = np.zeros(n, dtype=float)
        for i in range(n):
            cat = merchant_cats[i]
            avg = avg_amounts[cat]
            # log-normal around the category average
            amounts[i] = max(
                1.0,
                rng.lognormal(mean=np.log(avg), sigma=0.5),
            )

        # direct deposits tend to be larger (salary-like)
        is_deposit = txn_types == "direct_deposit"
        amounts[is_deposit] = rng.lognormal(
            mean=np.log(3500), sigma=0.3, size=is_deposit.sum()
        )

        # ATM withdrawals capped at 500
        is_atm = txn_types == "atm_withdrawal"
        amounts[is_atm] = np.clip(amounts[is_atm], 20, 500)

        amounts = amounts.round(2)

        # --- weekend boost: duplicate some transactions on weekends ---
        day_of_week = txn_dates.dayofweek  # 5=Sat, 6=Sun
        weekend_mask = day_of_week >= 5
        # scale up amounts slightly on weekends (more spending)
        amounts[weekend_mask] = (amounts[weekend_mask] * rng.uniform(1.0, 1.5, size=weekend_mask.sum())).round(2)

        # --- payday deposits: boost on 1st and 15th ---
        day_of_month = txn_dates.day
        is_payday = np.isin(day_of_month, [1, 15])
        payday_mask = is_payday & is_deposit
        amounts[payday_mask] = (amounts[payday_mask] * rng.uniform(1.0, 1.2, size=payday_mask.sum())).round(2)

        # --- account IDs (one or more per customer) ---
        account_ids = np.array([f"ACC{str(rng.integers(1, 999999)).zfill(6)}" for _ in range(n)])

        # --- running balance (simplified) ---
        balances = np.cumsum(np.where(
            np.isin(txn_types, ["credit", "direct_deposit", "transfer"]),
            amounts,
            -amounts,
        )).round(2)

        # --- descriptions ---
        descriptions = self._generate_descriptions(txn_types, merchant_cats, rng)

        df = pd.DataFrame({
            "transaction_id": [f"TXN{str(i).zfill(10)}" for i in range(1, n + 1)],
            "customer_id": cust_ids,
            "account_id": account_ids,
            "transaction_date": txn_dates,
            "transaction_type": txn_types,
            "amount": amounts,
            "balance_after": balances,
            "merchant_category": merchant_cats,
            "description": descriptions,
            "product_id": prod_ids,
        })

        logger.info("Generated %d transactions.", len(df))
        return df

    @staticmethod
    def _generate_descriptions(
        txn_types: np.ndarray,
        merchant_cats: np.ndarray,
        rng: np.random.Generator,
    ) -> List[str]:
        """Build human-readable transaction descriptions."""
        verb_map = {
            "debit": "Purchase at",
            "credit": "Refund from",
            "transfer": "Transfer to/from",
            "atm_withdrawal": "ATM withdrawal at",
            "direct_deposit": "Direct deposit from",
        }
        return [
            f"{verb_map.get(t, t)} {m}"
            for t, m in zip(txn_types, merchant_cats)
        ]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets and return as a dict.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: ``customers``, ``products``, ``transactions``.
        """
        logger.info("Generating full synthetic dataset …")
        customers = self.generate_customers()
        products = self.generate_products()
        transactions = self.generate_transactions(customers, products)

        logger.info(
            "Dataset ready — %d customers, %d products, %d transactions.",
            len(customers), len(products), len(transactions),
        )
        return {
            "customers": customers,
            "products": products,
            "transactions": transactions,
        }


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    """CLI helper: generate data and write to parquet files."""
    import argparse
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic bank data.")
    parser.add_argument("--customers", type=int, default=5000)
    parser.add_argument("--transactions", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/raw")
    args = parser.parse_args()

    gen = BankDataGenerator(
        n_customers=args.customers,
        n_transactions=args.transactions,
        seed=args.seed,
    )
    data = gen.generate_all()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, df in data.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Wrote %s → %s  (%d rows)", name, path, len(df))


if __name__ == "__main__":
    main()
