# Bank Transaction Data Warehouse & Analytical Layer

A production-grade data warehouse pipeline that generates synthetic bank transaction data, loads it into a **star-schema** warehouse, applies **SCD Type 2** history tracking on the customer dimension, and builds analytical views for churn prediction and cross-sell affinity analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI  (run.py)                            │
│   full │ generate │ load │ quality │ views                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                         │
│  create_tables → generate → load → quality → scd2 → views      │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Synthetic  │ │   SQLAlchemy │ │   Analytical │
│   Data Gen   │ │   Engine     │ │   Views      │
│  (NumPy)     │ │  (SQLite)    │ │  (SQL VIEWS) │
└──────────────┘ └──────────────┘ └──────┬───────┘
                                         │
                                         ▼
                                  ┌──────────────┐
                                  │   CSV Export  │
                                  │  (data/outputs│
                                  └──────────────┘
```

---

## Star Schema

```
                       ┌───────────────────┐
                       │    dim_date        │
                       │───────────────────│
                       │ date_sk (PK)      │
                       │ full_date         │
                       │ day / month / year│
                       │ quarter           │
                       │ day_of_week       │
                       │ is_weekend        │
                       └────────┬──────────┘
                                │
┌───────────────────┐           │           ┌───────────────────┐
│  dim_customer     │           │           │   dim_product     │
│───────────────────│           │           │───────────────────│
│ customer_sk (PK)  │     ┌─────┴─────┐     │ product_sk (PK)   │
│ customer_id       │     │   fact_    │     │ product_id        │
│ name / email      ├────►│transaction ├────►│ product_name      │
│ segment / income  │     │────────────│     │ category          │
│ effective/expiry   │     │ amount     │     └───────────────────┘
│ is_current        │     │ fee        │
└───────────────────┘     │ channel    │     ┌───────────────────┐
                          │ status     │     │   dim_branch      │
                          └─────┬──────┘     │───────────────────│
                                │            │ branch_sk (PK)    │
                                └───────────►│ branch_id         │
                                             │ city / state      │
                                             │ region            │
                                             └───────────────────┘
```

---

## SCD Type 2 — Slowly Changing Dimensions

The `dim_customer` dimension tracks historical attribute changes using **SCD Type 2**:

| Column | Purpose |
|---|---|
| `customer_sk` | Auto-incrementing surrogate key |
| `effective_date` | Date when this version became active |
| `expiry_date` | Date when this version was superseded (`NULL` = current) |
| `is_current` | `1` if this is the latest version, `0` otherwise |

**Tracked columns** (configurable in `default.yaml`):
`customer_name`, `email`, `phone`, `address`, `city`, `state`, `income_bracket`, `customer_segment`

**Behaviour during pipeline run:**
1. ~5 % of current customer rows are randomly mutated.
2. Existing rows are expired (`is_current = 0`, `expiry_date = today`).
3. New versioned rows are inserted (`is_current = 1`, `effective_date = tomorrow`).

This preserves a full audit trail of every attribute change over time.

---

## Quick Start

```bash
# 1. Clone / enter the project
cd Bank-Data-Warehouse

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline
python data_warehouse/run.py full

# — or run individual stages —
python data_warehouse/run.py generate
python data_warehouse/run.py quality
python data_warehouse/run.py views
python data_warehouse/run.py load
```

### CLI Options

| Flag | Description |
|---|---|
| `-c`, `--config` | Path to a custom YAML config (default: `configs/default.yaml`) |
| `-v`, `--verbose` | Enable `DEBUG`-level logging |
| `full` | Run every stage end-to-end |
| `generate` | Generate synthetic data and load into warehouse |
| `load` | Regenerate source data and ETL-load |
| `quality` | Run data-quality checks only |
| `views` | Rebuild analytical views and export CSVs |

---

## Output Files

All exports are written to `data_warehouse/data/outputs/`:

| File | Description |
|---|---|
| `v_churn_risk.csv` | Per-customer churn status, days since last txn, activity windows |
| `v_cross_sell_affinity.csv` | Product-pair support, lift, and strong-affinity flag |
| `v_monthly_revenue.csv` | Monthly revenue by product category and channel |
| `dim_customer.csv` | Full SCD2 customer dimension snapshot |
| `dim_product.csv` | Product catalogue |
| `dim_date.csv` | Date dimension |
| `dim_branch.csv` | Branch dimension |

The SQLite warehouse itself lives at `data_warehouse/data/warehouse.db`.

---

## Configuration

All tunables live in `data_warehouse/configs/default.yaml`:

```yaml
database:
  url: sqlite:///data/warehouse.db

data:
  n_customers: 5000
  n_transactions: 500000
  seed: 42

quality_checks:
  customers:
    not_null: [customer_id, customer_name, email]
    unique: [customer_id]
    values_in_set:
      gender: [Male, Female, Non-binary]
      customer_segment: [Premium, Regular, Basic]
  transactions:
    not_null: [transaction_id, customer_id, product_id, amount, transaction_date]
    unique: [transaction_id]
    range:
      amount: [0.01, 1000000]
  products:
    not_null: [product_id, product_name]
    unique: [product_id]

scd_type2:
  tracked_columns: [customer_name, email, phone, address, city, state, income_bracket, customer_segment]

analytical_views:
  churn:
    inactive_threshold_days: 60
    windows: [30, 60, 90]
  cross_sell:
    min_support: 0.01
    affinity_threshold: 0.1
```

---

## Dependencies

| Package | Version |
|---|---|
| `numpy` | >= 1.24.0 |
| `pandas` | >= 2.0.0 |
| `sqlalchemy` | >= 2.0.0 |
| `pyyaml` | >= 6.0 |

---

## Project Structure

```
Bank-Data-Warehouse/
├── data_warehouse/
│   ├── __init__.py              # Package metadata
│   ├── run.py                   # CLI entry point
│   ├── configs/
│   │   └── default.yaml         # All configuration
│   ├── src/
│   │   ├── __init__.py
│   │   └── pipeline.py          # Full ETL + quality + SCD2 + views
│   └── data/
│       ├── warehouse.db         # SQLite warehouse (auto-created)
│       └── outputs/             # CSV exports (auto-created)
├── requirements.txt
└── README.md
```

---

## License

MIT
