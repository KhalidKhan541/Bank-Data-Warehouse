import logging
from datetime import datetime, date, timedelta

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import Session

from models import DimDate, DimCustomer, DimProduct, FactTransaction, Base

logger = logging.getLogger(__name__)


class ETLPipeline:
    """ETL pipeline for loading data into the star schema."""

    def __init__(self, engine, config: dict):
        self.engine = engine
        self.config = config
        self.logger = logging.getLogger(__name__)

    # ── Extract ────────────────────────────────────────────────────────────────

    def extract_transactions(self, source_path: str) -> pd.DataFrame:
        """Extract raw transactions from CSV/database."""
        self.logger.info("Extracting transactions from %s", source_path)
        try:
            df = pd.read_csv(source_path, parse_dates=["transaction_date", "created_at"])
            self.logger.info("Extracted %d transaction rows", len(df))
            return df
        except FileNotFoundError:
            self.logger.error("Transaction file not found: %s", source_path)
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            self.logger.warning("Transaction file is empty: %s", source_path)
            return pd.DataFrame()

    def extract_customers(self, source_path: str) -> pd.DataFrame:
        """Extract customer data from source system."""
        self.logger.info("Extracting customers from %s", source_path)
        try:
            df = pd.read_csv(source_path, parse_dates=["date_of_birth", "account_open_date"])
            self.logger.info("Extracted %d customer rows", len(df))
            return df
        except FileNotFoundError:
            self.logger.error("Customer file not found: %s", source_path)
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            self.logger.warning("Customer file is empty: %s", source_path)
            return pd.DataFrame()

    def extract_products(self, source_path: str) -> pd.DataFrame:
        """Extract product catalog."""
        self.logger.info("Extracting products from %s", source_path)
        try:
            df = pd.read_csv(source_path)
            self.logger.info("Extracted %d product rows", len(df))
            return df
        except FileNotFoundError:
            self.logger.error("Product file not found: %s", source_path)
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            self.logger.warning("Product file is empty: %s", source_path)
            return pd.DataFrame()

    # ── Transform ──────────────────────────────────────────────────────────────

    def transform_dim_date(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate date dimension table for date range."""
        self.logger.info("Generating date dimension from %s to %s", start_date, end_date)
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        days = (end - start).days + 1
        dates = [start + timedelta(days=i) for i in range(days)]

        records = []
        for d in dates:
            records.append({
                "date_key": int(d.strftime("%Y%m%d")),
                "full_date": d,
                "day_of_week": d.strftime("%A"),
                "day_of_month": d.day,
                "month": d.month,
                "month_name": d.strftime("%B"),
                "quarter": (d.month - 1) // 3 + 1,
                "year": d.year,
                "is_weekend": d.weekday() >= 5,
                "is_holiday": False,
                "fiscal_year": d.year if d.month >= 4 else d.year - 1,
                "fiscal_quarter": (d.month - 1) // 3 + 1,
                "week_of_year": d.isocalendar()[1],
            })
        df = pd.DataFrame(records)
        self.logger.info("Generated %d date rows", len(df))
        return df

    def transform_dim_product(self, raw_products: pd.DataFrame) -> pd.DataFrame:
        """Transform and deduplicate products."""
        if raw_products.empty:
            self.logger.warning("No products to transform")
            return raw_products
        before = len(raw_products)
        raw_products = raw_products.drop_duplicates(subset=["product_id"], keep="last")
        self.logger.info("Deduplicated products: %d → %d rows", before, len(raw_products))
        return raw_products.reset_index(drop=True)

    def transform_dim_customer(self, raw_customers: pd.DataFrame) -> pd.DataFrame:
        """SCD Type 2 transformation for customer dimension.

        - New customers: insert with is_current=True, version=1
        - Changed customers: expire old record, insert new with version+1
        - Unchanged customers: keep as-is
        Returns only changed/new records for upsert.
        """
        if raw_customers.empty:
            self.logger.warning("No customers to transform")
            return pd.DataFrame()

        self.logger.info("Running SCD Type 2 transformation for %d customers", len(raw_customers))
        raw_customers = raw_customers.drop_duplicates(subset=["customer_id"], keep="last")

        scd_columns = [
            "customer_name", "email", "phone", "address", "city", "state",
            "zip_code", "country", "date_of_birth", "gender", "income_bracket",
            "customer_segment", "account_open_date",
        ]

        new_records = []
        with Session(self.engine) as session:
            for _, row in raw_customers.iterrows():
                customer_id = row["customer_id"]
                existing = (
                    session.query(DimCustomer)
                    .filter(DimCustomer.customer_id == customer_id, DimCustomer.is_current == True)
                    .first()
                )

                if existing is None:
                    new_records.append({
                        "customer_id": customer_id,
                        **{col: row.get(col) for col in scd_columns},
                        "effective_date": date.today(),
                        "expiry_date": None,
                        "is_current": True,
                        "version": 1,
                    })
                else:
                    changed = any(
                        getattr(existing, col) != row.get(col)
                        for col in scd_columns
                    )
                    if changed:
                        new_records.append({
                            "customer_id": customer_id,
                            **{col: row.get(col) for col in scd_columns},
                            "effective_date": date.today(),
                            "expiry_date": None,
                            "is_current": True,
                            "version": existing.version + 1,
                        })

        df = pd.DataFrame(new_records)
        self.logger.info("SCD2 produced %d new/changed customer records", len(df))
        return df

    def transform_fact_transactions(self, raw_transactions: pd.DataFrame) -> pd.DataFrame:
        """Transform transactions with surrogate key lookups."""
        if raw_transactions.empty:
            self.logger.warning("No transactions to transform")
            return pd.DataFrame()

        self.logger.info("Transforming %d transactions", len(raw_transactions))

        df = raw_transactions.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"]).dt.date
        df["date_key"] = df["transaction_date"].apply(lambda d: int(d.strftime("%Y%m%d")))

        # ── Data quality: drop rows with null required fields ──────────────────
        required = ["transaction_id", "customer_id", "product_id", "amount", "transaction_date"]
        before = len(df)
        df = df.dropna(subset=required)
        if len(df) < before:
            self.logger.warning("Dropped %d transactions with null required fields", before - len(df))

        # ── Data quality: valid dates ──────────────────────────────────────────
        valid_dates = df["date_key"].between(19000101, 29991231)
        invalid = (~valid_dates).sum()
        if invalid:
            self.logger.warning("Dropped %d transactions with invalid dates", invalid)
            df = df[valid_dates]

        # ── Surrogate key lookups ──────────────────────────────────────────────
        with Session(self.engine) as session:
            # date_key → already computed above
            # customer_id → customer_sk
            customer_map = {
                c.customer_id: c.customer_sk
                for c in session.query(DimCustomer.customer_id, DimCustomer.customer_sk)
                    .filter(DimCustomer.is_current == True).all()
            }
            # product_id → product_sk
            product_map = {
                p.product_id: p.product_sk
                for p in session.query(DimProduct.product_id, DimProduct.product_sk).all()
            }

        df["customer_sk"] = df["customer_id"].map(customer_map)
        df["product_sk"] = df["product_id"].map(product_map)

        missing_customers = df["customer_sk"].isna().sum()
        missing_products = df["product_sk"].isna().sum()
        if missing_customers:
            self.logger.warning("%d transactions reference missing customers – dropping", missing_customers)
        if missing_products:
            self.logger.warning("%d transactions reference missing products – dropping", missing_products)

        df = df.dropna(subset=["customer_sk", "product_sk"])
        df["customer_sk"] = df["customer_sk"].astype(int)
        df["product_sk"] = df["product_sk"].astype(int)

        self.logger.info("Transformed %d valid fact rows", len(df))
        return df

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_dim_date(self, dim_date: pd.DataFrame):
        """Load date dimension."""
        if dim_date.empty:
            self.logger.info("No date rows to load")
            return
        self.logger.info("Loading %d date rows", len(dim_date))
        records = dim_date.to_dict(orient="records")
        with Session(self.engine) as session:
            existing_keys = {
                r[0]
                for r in session.query(DimDate.date_key).all()
            }
            new_records = [r for r in records if r["date_key"] not in existing_keys]
            if new_records:
                session.bulk_insert_mappings(DimDate, new_records)
                session.commit()
                self.logger.info("Inserted %d new date rows", len(new_records))
            else:
                self.logger.info("Date dimension already up to date")

    def load_dim_product(self, dim_product: pd.DataFrame):
        """Load product dimension."""
        if dim_product.empty:
            self.logger.info("No product rows to load")
            return
        self.logger.info("Loading %d product rows", len(dim_product))
        with Session(self.engine) as session:
            existing_ids = {
                r[0]
                for r in session.query(DimProduct.product_id).all()
            }
            new_records = []
            for _, row in dim_product.iterrows():
                if row["product_id"] not in existing_ids:
                    new_records.append(row.to_dict())
            if new_records:
                session.bulk_insert_mappings(DimProduct, new_records)
                session.commit()
                self.logger.info("Inserted %d new product rows", len(new_records))
            else:
                self.logger.info("All products already exist")

    def load_dim_customer_scd2(self, new_customers: pd.DataFrame):
        """Load customer dimension with SCD Type 2 logic.

        1. Find existing current records
        2. Compare with new data for changes
        3. Expire changed records (set expiry_date, is_current=False)
        4. Insert new versions
        """
        if new_customers.empty:
            self.logger.info("No customer changes to load")
            return

        self.logger.info("Loading %d new/changed customer records (SCD2)", len(new_customers))
        today = date.today()

        with Session(self.engine) as session:
            for _, row in new_customers.iterrows():
                customer_id = row["customer_id"]
                version = row["version"]

                if version > 1:
                    old_record = (
                        session.query(DimCustomer)
                        .filter(
                            DimCustomer.customer_id == customer_id,
                            DimCustomer.is_current == True,
                        )
                        .first()
                    )
                    if old_record:
                        old_record.expiry_date = today
                        old_record.is_current = False
                        self.logger.debug("Expired customer %s version %d", customer_id, old_record.version)

                new_record = DimCustomer(
                    customer_id=customer_id,
                    customer_name=row.get("customer_name"),
                    email=row.get("email"),
                    phone=row.get("phone"),
                    address=row.get("address"),
                    city=row.get("city"),
                    state=row.get("state"),
                    zip_code=row.get("zip_code"),
                    country=row.get("country"),
                    date_of_birth=row.get("date_of_birth"),
                    gender=row.get("gender"),
                    income_bracket=row.get("income_bracket"),
                    customer_segment=row.get("customer_segment"),
                    account_open_date=row.get("account_open_date"),
                    effective_date=today,
                    expiry_date=None,
                    is_current=True,
                    version=version,
                )
                session.add(new_record)

            session.commit()
            self.logger.info("Committed SCD2 customer changes")

    def load_fact_transactions(self, facts: pd.DataFrame):
        """Load transaction facts."""
        if facts.empty:
            self.logger.info("No fact rows to load")
            return
        self.logger.info("Loading %d fact rows", len(facts))

        cols = [
            "transaction_id", "date_key", "customer_sk", "product_sk",
            "amount", "balance_after", "transaction_type", "transaction_category",
            "channel", "merchant", "merchant_category", "created_at", "is_fraudulent",
        ]
        records = facts[[c for c in cols if c in facts.columns]].to_dict(orient="records")

        with Session(self.engine) as session:
            existing_ids = {
                r[0]
                for r in session.query(FactTransaction.transaction_id).all()
            }
            new_records = [r for r in records if r["transaction_id"] not in existing_ids]
            if new_records:
                session.bulk_insert_mappings(FactTransaction, new_records)
                session.commit()
                self.logger.info("Inserted %d new fact rows", len(new_records))
            else:
                self.logger.info("All transactions already loaded")

    # ── Pipeline runners ───────────────────────────────────────────────────────

    def run_full_load(self, transactions_path: str, customers_path: str, products_path: str):
        """Run full ETL pipeline."""
        self.logger.info("=== Starting full ETL load ===")
        start = self.config.get("date_range_start", "2020-01-01")
        end = self.config.get("date_range_end", date.today().isoformat())

        raw_tx = self.extract_transactions(transactions_path)
        raw_cust = self.extract_customers(customers_path)
        raw_prod = self.extract_products(products_path)

        dim_date = self.transform_dim_date(start, end)
        dim_product = self.transform_dim_product(raw_prod)
        dim_customer = self.transform_dim_customer(raw_cust)

        self.load_dim_date(dim_date)
        self.load_dim_product(dim_product)
        self.load_dim_customer_scd2(dim_customer)

        facts = self.transform_fact_transactions(raw_tx)
        self.load_fact_transactions(facts)

        self.logger.info("=== Full ETL load complete ===")

    def run_incremental_load(self, transactions_path: str, customers_path: str):
        """Run incremental ETL (only new transactions + SCD2 for customers)."""
        self.logger.info("=== Starting incremental ETL load ===")

        raw_tx = self.extract_transactions(transactions_path)
        raw_cust = self.extract_customers(customers_path)

        dim_customer = self.transform_dim_customer(raw_cust)
        self.load_dim_customer_scd2(dim_customer)

        facts = self.transform_fact_transactions(raw_tx)
        self.load_fact_transactions(facts)

        self.logger.info("=== Incremental ETL load complete ===")
