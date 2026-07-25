"""Main ETL pipeline orchestrator.

Coordinates the full lifecycle: schema creation → data generation →
quality checks → warehouse load → SCD2 processing → analytical views.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and return the YAML configuration."""
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _get_engine(cfg: dict[str, Any]) -> Engine:
    """Create a SQLAlchemy engine from config."""
    db_path = Path(__file__).resolve().parent.parent / "data"
    db_path.mkdir(parents=True, exist_ok=True)
    url = cfg["database"]["url"]
    if "sqlite" in url:
        rel = url.split("///", 1)[-1]
        abs_path = db_path / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{abs_path}"
    engine = create_engine(url, echo=False)
    logger.info("Engine created: %s", url)
    return engine


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

DDL = """
-- Dim: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_sk       INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id       TEXT    NOT NULL,
    customer_name     TEXT,
    email             TEXT,
    phone             TEXT,
    address           TEXT,
    city              TEXT,
    state             TEXT,
    zip_code          TEXT,
    gender            TEXT,
    date_of_birth     TEXT,
    income_bracket    TEXT,
    customer_segment  TEXT,
    effective_date    TEXT NOT NULL,
    expiry_date       TEXT,
    is_current        INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_dim_customer_id ON dim_customer(customer_id);

-- Dim: Product
CREATE TABLE IF NOT EXISTS dim_product (
    product_sk    INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    TEXT    NOT NULL UNIQUE,
    product_name  TEXT,
    category      TEXT,
    subcategory   NULL
);

-- Dim: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_sk       INTEGER PRIMARY KEY,
    full_date     TEXT NOT NULL UNIQUE,
    day           INTEGER,
    month         INTEGER,
    year          INTEGER,
    quarter       INTEGER,
    day_of_week   INTEGER,
    is_weekend    INTEGER
);

-- Dim: Branch
CREATE TABLE IF NOT EXISTS dim_branch (
    branch_sk     INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id     TEXT    NOT NULL UNIQUE,
    branch_name   TEXT,
    city          TEXT,
    state         TEXT,
    region        TEXT
);

-- Fact: Transaction
CREATE TABLE IF NOT EXISTS fact_transaction (
    transaction_sk     INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id     TEXT    NOT NULL,
    customer_sk        INTEGER NOT NULL,
    product_sk         INTEGER NOT NULL,
    date_sk            INTEGER NOT NULL,
    branch_sk          INTEGER,
    amount             REAL    NOT NULL,
    fee                REAL    DEFAULT 0.0,
    transaction_type   TEXT,
    channel            TEXT,
    status             TEXT,
    created_at         TEXT,
    FOREIGN KEY (customer_sk) REFERENCES dim_customer(customer_sk),
    FOREIGN KEY (product_sk)  REFERENCES dim_product(product_sk),
    FOREIGN KEY (date_sk)     REFERENCES dim_date(date_sk),
    FOREIGN KEY (branch_sk)   REFERENCES dim_branch(branch_sk)
);
CREATE INDEX IF NOT EXISTS idx_fact_cust ON fact_transaction(customer_sk);
CREATE INDEX IF NOT EXISTS idx_fact_date ON fact_transaction(date_sk);

-- Staging
CREATE TABLE IF NOT EXISTS stg_customers   (LIKE dim_customer    EXCLUDING IDENTITY);
CREATE TABLE IF NOT EXISTS stg_transactions (LIKE fact_transaction EXCLUDING IDENTITY);
"""


def create_tables(engine: Engine) -> None:
    """Create the star-schema tables in the warehouse."""
    logger.info("Creating warehouse tables …")
    with engine.begin() as conn:
        for statement in DDL.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("All tables created.")


# ---------------------------------------------------------------------------
# Data Generation
# ---------------------------------------------------------------------------

def _generate_dim_date(start: str = "2020-01-01", end: str = "2026-12-31") -> pd.DataFrame:
    """Generate a date dimension spanning *start* → *end*."""
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({
        "date_sk": dates.strftime("%Y%m%d").astype(int),
        "full_date": dates.strftime("%Y-%m-%d"),
        "day": dates.day,
        "month": dates.month,
        "year": dates.year,
        "quarter": dates.quarter,
        "day_of_week": dates.dayofweek,
        "is_weekend": (dates.dayofweek >= 5).astype(int),
    })
    return df.drop_duplicates(subset="date_sk").reset_index(drop=True)


def _generate_dim_branch(n: int = 50, rng: np.random.Generator = np.random.default_rng(42)) -> pd.DataFrame:
    """Generate synthetic branch data."""
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
              "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin"]
    states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "TX"]
    regions = ["Northeast", "West", "Midwest", "South", "West",
               "Northeast", "South", "West", "South", "South"]
    rows = []
    for i in range(1, n + 1):
        idx = rng.integers(0, len(cities))
        rows.append({
            "branch_id": f"BR{i:04d}",
            "branch_name": f"Branch {i:04d}",
            "city": cities[idx],
            "state": states[idx],
            "region": regions[idx],
        })
    return pd.DataFrame(rows)


def _generate_dim_product() -> pd.DataFrame:
    """Generate a fixed set of banking products."""
    products = [
        ("P001", "Checking Account", "Deposit", "Checking"),
        ("P002", "Savings Account", "Deposit", "Savings"),
        ("P003", "Certificate of Deposit", "Deposit", "Term Deposit"),
        ("P004", "Personal Loan", "Loan", "Consumer"),
        ("P005", "Home Mortgage", "Loan", "Mortgage"),
        ("P006", "Auto Loan", "Loan", "Consumer"),
        ("P007", "Credit Card", "Card", "Unsecured"),
        ("P008", "Debit Card", "Card", "Debit"),
        ("P009", "Wire Transfer", "Payment", "Domestic"),
        ("P010", "International Wire", "Payment", "International"),
        ("P011", "Money Market", "Deposit", "Money Market"),
        ("P012", "Student Loan", "Loan", "Consumer"),
        ("P013", "Business Loan", "Loan", "Commercial"),
        ("P014", "Insurance Premium", "Insurance", "Life"),
        ("P015", "Investment Account", "Investment", "Brokerage"),
    ]
    return pd.DataFrame(products, columns=["product_id", "product_name", "category", "subcategory"])


def generate_data(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Generate all synthetic source data and return as a dict of DataFrames."""
    rng = np.random.default_rng(cfg["data"]["seed"])
    n_cust = cfg["data"]["n_customers"]
    n_txn = cfg["data"]["n_transactions"]

    logger.info("Generating %d customers …", n_cust)
    first_names = [f"First_{i}" for i in range(n_cust)]
    last_names = [f"Last_{i}" for i in range(n_cust)]
    genders = rng.choice(["Male", "Female", "Non-binary"], n_cust, p=[0.45, 0.45, 0.10])
    segments = rng.choice(["Premium", "Regular", "Basic"], n_cust, p=[0.15, 0.50, 0.35])
    incomes = rng.choice(["<30k", "30k-60k", "60k-100k", "100k-200k", ">200k"], n_cust)
    customers = pd.DataFrame({
        "customer_id": [f"CUST{i:06d}" for i in range(n_cust)],
        "customer_name": [f"{fn} {ln}" for fn, ln in zip(first_names, last_names)],
        "email": [f"customer{i}@example.com" for i in range(n_cust)],
        "phone": [f"+1-555-{i:04d}" for i in rng.integers(0, 9999, n_cust)],
        "address": [f"{i} Main St" for i in range(1, n_cust + 1)],
        "city": rng.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n_cust),
        "state": rng.choice(["NY", "CA", "IL", "TX", "AZ"], n_cust),
        "zip_code": [str(rng.integers(10000, 99999)) for _ in range(n_cust)],
        "gender": genders,
        "date_of_birth": pd.date_range("1950-01-01", periods=n_cust, freq="3D").strftime("%Y-%m-%d"),
        "income_bracket": incomes,
        "customer_segment": segments,
    })

    products = _generate_dim_product()
    branches = _generate_dim_branch(50, rng)
    dates = _generate_dim_date()

    logger.info("Generating %d transactions …", n_txn)
    cust_ids = customers["customer_id"].values
    prod_ids = products["product_id"].values
    branch_ids = branches["branch_id"].values
    date_sks = dates["date_sk"].values

    txn_amounts = np.round(rng.lognormal(mean=5.0, sigma=2.0, size=n_txn), 2)
    txn_amounts = np.clip(txn_amounts, 0.01, 1_000_000)

    transactions = pd.DataFrame({
        "transaction_id": [f"TXN{i:08d}" for i in range(n_txn)],
        "customer_id": rng.choice(cust_ids, n_txn),
        "product_id": rng.choice(prod_ids, n_txn, p=np.array([0.20, 0.15, 0.05, 0.08, 0.06, 0.06, 0.12, 0.10, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.01])),
        "branch_id": rng.choice(branch_ids, n_txn),
        "amount": txn_amounts,
        "fee": np.round(rng.uniform(0, 25, n_txn), 2),
        "transaction_type": rng.choice(["Debit", "Credit"], n_txn),
        "channel": rng.choice(["Branch", "ATM", "Online", "Mobile"], n_txn, p=[0.15, 0.10, 0.40, 0.35]),
        "status": rng.choice(["Completed", "Pending", "Failed"], n_txn, p=[0.90, 0.07, 0.03]),
        "transaction_date": rng.choice(date_sks, n_txn),
        "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    logger.info("Data generation complete.")
    return {
        "dim_customer": customers,
        "dim_product": products,
        "dim_branch": branches,
        "dim_date": dates,
        "fact_transaction": transactions,
    }


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------

def run_quality_checks(engine: Engine, cfg: dict[str, Any]) -> dict[str, list[str]]:
    """Execute data-quality checks defined in config. Returns failures."""
    checks_cfg = cfg.get("quality_checks", {})
    failures: dict[str, list[str]] = {}

    for table_name, rules in checks_cfg.items():
        table_failures: list[str] = []
        try:
            df = pd.read_sql_table(table_name, engine)
        except Exception as exc:
            table_failures.append(f"Cannot read table {table_name}: {exc}")
            failures[table_name] = table_failures
            continue

        for col in rules.get("not_null", []):
            if col in df.columns:
                null_count = int(df[col].isna().sum())
                if null_count:
                    table_failures.append(f"NOT NULL violation on '{col}': {null_count} nulls")

        for col in rules.get("unique", []):
            if col in df.columns:
                dup_count = int(df[col].duplicated().sum())
                if dup_count:
                    table_failures.append(f"UNIQUE violation on '{col}': {dup_count} duplicates")

        for col, allowed in rules.get("values_in_set", {}).items():
            if col in df.columns:
                invalid = set(df[col].dropna().unique()) - set(allowed)
                if invalid:
                    table_failures.append(f"VALUES_IN_SET violation on '{col}': unexpected {invalid}")

        for col, (lo, hi) in rules.get("range", {}).items():
            if col in df.columns:
                out = int(((df[col] < lo) | (df[col] > hi)).sum())
                if out:
                    table_failures.append(f"RANGE violation on '{col}': {out} values outside [{lo}, {hi}]")

        if table_failures:
            failures[table_name] = table_failures
            logger.warning("Quality issues in %s: %s", table_name, table_failures)
        else:
            logger.info("Quality checks passed for %s.", table_name)

    return failures


# ---------------------------------------------------------------------------
# ETL Load
# ---------------------------------------------------------------------------

def load_to_warehouse(engine: Engine, data: dict[str, pd.DataFrame]) -> None:
    """Insert generated data into the warehouse dimensions and fact table."""
    logger.info("Loading data into warehouse …")

    # Dim: Date
    dates = data["dim_date"]
    dates.to_sql("dim_date", engine, if_exists="append", index=False)
    logger.info("  dim_date: %d rows", len(dates))

    # Dim: Branch
    branches = data["dim_branch"]
    branches.to_sql("dim_branch", engine, if_exists="append", index=False)
    logger.info("  dim_branch: %d rows", len(branches))

    # Dim: Product
    products = data["dim_product"]
    products.to_sql("dim_product", engine, if_exists="append", index=False)
    logger.info("  dim_product: %d rows", len(products))

    # Dim: Customer (initial load — all rows effective today, is_current=1)
    customers = data["dim_customer"].copy()
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    customers["effective_date"] = today
    customers["expiry_date"] = None
    customers["is_current"] = 1
    customers.to_sql("dim_customer", engine, if_exists="append", index=False)
    logger.info("  dim_customer: %d rows", len(customers))

    # Fact: Transaction — resolve surrogate keys
    txn = data["fact_transaction"].copy()
    cust_map = pd.read_sql("SELECT customer_sk, customer_id FROM dim_customer", engine)
    prod_map = pd.read_sql("SELECT product_sk, product_id FROM dim_product", engine)
    branch_map = pd.read_sql("SELECT branch_sk, branch_id FROM dim_branch", engine)

    txn = txn.merge(cust_map, on="customer_id", how="left")
    txn = txn.rename(columns={"customer_sk": "customer_sk"})
    txn = txn.merge(prod_map, on="product_id", how="left")
    txn = txn.merge(branch_map, on="branch_id", how="left")
    txn["date_sk"] = txn["transaction_date"].astype(int)

    fact_cols = [
        "transaction_id", "customer_sk", "product_sk", "date_sk", "branch_sk",
        "amount", "fee", "transaction_type", "channel", "status", "created_at",
    ]
    txn[fact_cols].to_sql("fact_transaction", engine, if_exists="append", index=False)
    logger.info("  fact_transaction: %d rows", len(txn))


# ---------------------------------------------------------------------------
# SCD Type 2
# ---------------------------------------------------------------------------

def apply_scd2(engine: Engine, cfg: dict[str, Any]) -> None:
    """Simulate SCD Type 2 by expiring changed customer rows and inserting new ones."""
    tracked = cfg.get("scd_type2", {}).get("tracked_columns", [])
    if not tracked:
        logger.info("No SCD2 columns configured — skipping.")
        return

    logger.info("Applying SCD Type 2 for columns: %s", tracked)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    tomorrow = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    current = pd.read_sql(
        "SELECT * FROM dim_customer WHERE is_current = 1", engine
    )
    if current.empty:
        logger.info("No current customer rows — skipping SCD2.")
        return

    # Simulate attribute changes for ~5 % of customers
    rng = np.random.default_rng(cfg["data"]["seed"])
    n_change = max(1, int(len(current) * 0.05))
    change_idx = rng.choice(current.index, size=n_change, replace=False)
    changed = current.loc[change_idx].copy()

    # Mutate tracked fields
    for col in tracked:
        if col in changed.columns:
            changed[col] = changed[col] + " [upd]"

    # Expire old rows
    expire_stmt = text(
        "UPDATE dim_customer SET expiry_date = :exp, is_current = 0 "
        "WHERE customer_sk = :sk"
    )
    insert_stmt = text(
        "INSERT INTO dim_customer "
        "(customer_id, customer_name, email, phone, address, city, state, "
        "zip_code, gender, date_of_birth, income_bracket, customer_segment, "
        "effective_date, expiry_date, is_current) "
        "VALUES (:customer_id, :customer_name, :email, :phone, :address, :city, "
        ":state, :zip_code, :gender, :date_of_birth, :income_bracket, "
        ":customer_segment, :effective_date, NULL, 1)"
    )

    with engine.begin() as conn:
        for _, row in changed.iterrows():
            conn.execute(expire_stmt, {"exp": today, "sk": int(row["customer_sk"])})
            new_row = row.to_dict()
            new_row["effective_date"] = tomorrow
            # Remove auto-generated key so it re-seeds
            new_row.pop("customer_sk", None)
            # Remove None values so SQLAlchemy treats them as NULL
            new_row = {k: v for k, v in new_row.items() if v is not None}
            conn.execute(insert_stmt, new_row)

    logger.info("SCD2: expired %d rows, inserted %d new versions.", n_change, n_change)


# ---------------------------------------------------------------------------
# Analytical Views
# ---------------------------------------------------------------------------

def create_analytical_views(engine: Engine, cfg: dict[str, Any]) -> None:
    """Create materialised analytical views for churn and cross-sell."""
    views_cfg = cfg.get("analytical_views", {})

    churn_cfg = views_cfg.get("churn", {})
    inactive_days = churn_cfg.get("inactive_threshold_days", 60)
    windows = churn_cfg.get("windows", [30, 60, 90])

    logger.info("Creating analytical views …")

    with engine.begin() as conn:
        # ---- Churn Risk ----
        windowClauses = []
        for w in windows:
            windowClauses.append(
                f"SUM(CASE WHEN d.full_date >= STRFTIME('%Y-%m-%d', 'now', '-{w} days') "
                f"THEN 1 ELSE 0 END) AS txn_last_{w}d"
            )
        windowSql = ",\n            ".join(windowClauses)

        conn.execute(text(f"""
            CREATE VIEW IF NOT EXISTS v_churn_risk AS
            SELECT
                c.customer_sk,
                c.customer_id,
                c.customer_name,
                c.customer_segment,
                COUNT(t.transaction_sk)                            AS total_transactions,
                MAX(d.full_date)                                   AS last_transaction_date,
                CAST(JULIANDAY('now') - JULIANDAY(MAX(d.full_date)) AS INTEGER)
                                                                   AS days_since_last_txn,
                {windowSql},
                CASE
                    WHEN JULIANDAY('now') - JULIANDAY(MAX(d.full_date)) > {inactive_days}
                        THEN 'Churned'
                    WHEN JULIANDAY('now') - JULIANDAY(MAX(d.full_date)) > {inactive_days * 0.5}
                        THEN 'At Risk'
                    ELSE 'Active'
                END AS churn_status
            FROM dim_customer c
            LEFT JOIN fact_transaction t ON c.customer_sk = t.customer_sk
            LEFT JOIN dim_date d         ON t.date_sk     = d.date_sk
            WHERE c.is_current = 1
            GROUP BY c.customer_sk;
        """))
        logger.info("  v_churn_risk created.")

        # ---- Cross-sell Affinity ----
        min_support = views_cfg.get("cross_sell", {}).get("min_support", 0.01)
        affinity_thr = views_cfg.get("cross_sell", {}).get("affinity_threshold", 0.1)
        conn.execute(text(f"""
            CREATE VIEW IF NOT EXISTS v_cross_sell_affinity AS
            WITH customer_products AS (
                SELECT customer_sk, product_sk
                FROM fact_transaction
                GROUP BY customer_sk, product_sk
            ),
            product_counts AS (
                SELECT product_sk, COUNT(*) AS n
                FROM customer_products
                GROUP BY product_sk
            ),
            pair_counts AS (
                SELECT a.product_sk AS product_a,
                       b.product_sk AS product_b,
                       COUNT(*)     AS n_together
                FROM customer_products a
                JOIN customer_products b
                  ON a.customer_sk = b.customer_sk AND a.product_sk < b.product_sk
                GROUP BY a.product_sk, b.product_sk
            )
            SELECT
                pa.product_id   AS product_a_id,
                pa.product_name AS product_a_name,
                pb.product_id   AS product_b_id,
                pb.product_name AS product_b_name,
                pc.n_together,
                CAST(pc.n_together AS REAL) / ca.n AS support_a,
                CAST(pc.n_together AS REAL) / cb.n AS support_b,
                CASE
                    WHEN CAST(pc.n_together AS REAL) / ca.n > {affinity_thr}
                      OR CAST(pc.n_together AS REAL) / cb.n > {affinity_thr}
                        THEN 1 ELSE 0
                END AS strong_affinity
            FROM pair_counts pc
            JOIN product_counts ca ON pc.product_a = ca.product_sk
            JOIN product_counts cb ON pc.product_b = cb.product_sk
            JOIN dim_product pa    ON pc.product_a = pa.product_sk
            JOIN dim_product pb    ON pc.product_b = pb.product_sk
            WHERE CAST(pc.n_together AS REAL) / ca.n >= {min_support}
               OR CAST(pc.n_together AS REAL) / cb.n >= {min_support};
        """))
        logger.info("  v_cross_sell_affinity created.")

        # ---- Monthly Revenue ----
        conn.execute(text("""
            CREATE VIEW IF NOT EXISTS v_monthly_revenue AS
            SELECT
                d.year,
                d.month,
                p.category              AS product_category,
                t.channel,
                COUNT(*)                AS transaction_count,
                ROUND(SUM(t.amount), 2) AS total_amount,
                ROUND(SUM(t.fee), 2)    AS total_fees
            FROM fact_transaction t
            JOIN dim_date d     ON t.date_sk    = d.date_sk
            JOIN dim_product p  ON t.product_sk = p.product_sk
            GROUP BY d.year, d.month, p.category, t.channel;
        """))
        logger.info("  v_monthly_revenue created.")


# ---------------------------------------------------------------------------
# Export / Save Outputs
# ---------------------------------------------------------------------------

def save_outputs(engine: Engine) -> None:
    """Export analytical views and key tables to CSV for downstream consumption."""
    out_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    views = [
        "v_churn_risk",
        "v_cross_sell_affinity",
        "v_monthly_revenue",
    ]
    for view in views:
        try:
            df = pd.read_sql(f"SELECT * FROM {view}", engine)
            path = out_dir / f"{view}.csv"
            df.to_csv(path, index=False)
            logger.info("Exported %s → %s (%d rows)", view, path, len(df))
        except Exception as exc:
            logger.warning("Could not export %s: %s", view, exc)

    # Also export dimension snapshots
    for tbl in ["dim_customer", "dim_product", "dim_date", "dim_branch"]:
        try:
            df = pd.read_sql(f"SELECT * FROM {tbl}", engine)
            path = out_dir / f"{tbl}.csv"
            df.to_csv(path, index=False)
            logger.info("Exported %s → %s (%d rows)", tbl, path, len(df))
        except Exception as exc:
            logger.warning("Could not export %s: %s", tbl, exc)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str | Path | None = None, skip_generate: bool = False) -> None:
    """Execute the full data-warehouse pipeline end-to-end.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file.  Falls back to ``configs/default.yaml``.
    skip_generate:
        When *True*, skip data generation and use only the tables already in the DB.
    """
    t0 = time.perf_counter()
    cfg = _load_config(config_path)
    engine = _get_engine(cfg)

    logger.info("=== Pipeline START ===")

    # 1. Schema
    create_tables(engine)

    # 2. Generate
    data: dict[str, pd.DataFrame] = {}
    if not skip_generate:
        data = generate_data(cfg)
    else:
        logger.info("Skipping generation — loading existing data only.")

    # 3. Load
    load_to_warehouse(engine, data)

    # 4. Quality
    failures = run_quality_checks(engine, cfg)
    if failures:
        logger.error("Quality check failures:\n%s", yaml.dump(failures, default_flow_style=False))
        raise RuntimeError("Data quality checks failed — see log for details.")

    # 5. SCD2
    apply_scd2(engine, cfg)

    # 6. Views
    create_analytical_views(engine, cfg)

    # 7. Export
    save_outputs(engine)

    elapsed = time.perf_counter() - t0
    logger.info("=== Pipeline COMPLETE in %.2fs ===", elapsed)
