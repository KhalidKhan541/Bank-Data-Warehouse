import logging
from datetime import date, datetime, timedelta
from typing import Optional, Set, Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text, func, case, and_, extract
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class AnalyticalViews:
    """Create analytical views from the data warehouse.

    Generates ML-ready feature tables for churn prediction, cross-sell
    recommendation, fraud detection, and business analytics.
    """

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ── helpers ────────────────────────────────────────────────────────────────

    def _query_df(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a SQL statement and return a DataFrame."""
        logger.debug("Executing query:\n%s", sql)
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def _today(self) -> date:
        return date.today()

    # ── customer features ─────────────────────────────────────────────────────

    def customer_features_view(self) -> pd.DataFrame:
        """Create customer feature table for ML.

        Combines demographics from dim_customer, transaction aggregates
        from fact_transactions, product holdings, and behavioral features.

        Returns DataFrame with customer_sk as index.
        """
        sql = """
            SELECT
                c.customer_sk,
                c.customer_id,
                c.customer_name,
                c.gender,
                c.income_bracket,
                c.customer_segment,
                c.city,
                c.state,
                c.account_open_date,
                c.date_of_birth,
                -- demographics
                EXTRACT(YAGE FROM AGE(CURRENT_DATE, c.date_of_birth)) AS age,
                EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.account_open_date)) / 30.0 AS tenure_months,
                -- transaction aggregates
                COUNT(t.transaction_sk) AS total_transactions,
                SUM(CASE WHEN t.transaction_type = 'Credit' THEN t.amount ELSE 0 END) AS total_credits,
                SUM(CASE WHEN t.transaction_type = 'Debit' THEN t.amount ELSE 0 END) AS total_debits,
                AVG(t.amount) AS avg_transaction_amount,
                STDDEV(t.amount) AS stddev_transaction_amount,
                MAX(d.full_date) AS last_transaction_date,
                EXTRACT(DAYS FROM AGE(CURRENT_DATE, MAX(d.full_date))) AS days_since_last_transaction,
                -- product holdings
                COUNT(DISTINCT t.product_sk) AS product_count,
                -- channel usage
                COUNT(DISTINCT CASE WHEN t.channel = 'Online' THEN t.transaction_sk END) AS online_tx_count,
                COUNT(DISTINCT CASE WHEN t.channel = 'Mobile' THEN t.transaction_sk END) AS mobile_tx_count,
                COUNT(DISTINCT CASE WHEN t.channel = 'ATM' THEN t.transaction_sk END) AS atm_tx_count,
                COUNT(DISTINCT CASE WHEN t.channel = 'Branch' THEN t.transaction_sk END) AS branch_tx_count,
                -- balance
                AVG(t.balance_after) AS avg_balance
            FROM dim_customer c
            LEFT JOIN fact_transactions t ON c.customer_sk = t.customer_sk
            LEFT JOIN dim_date d ON t.date_key = d.date_key
            WHERE c.is_current = TRUE
            GROUP BY c.customer_sk, c.customer_id, c.customer_name, c.gender,
                     c.income_bracket, c.customer_segment, c.city, c.state,
                     c.account_open_date, c.date_of_birth
        """
        df = self._query_df(sql)
        if df.empty:
            logger.warning("customer_features_view: no data returned")
            return pd.DataFrame()

        df = df.set_index("customer_sk")
        for col in ["age", "tenure_months", "days_since_last_transaction",
                     "total_transactions", "total_credits", "total_debits",
                     "avg_transaction_amount", "stddev_transaction_amount",
                     "product_count", "avg_balance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info("customer_features_view: %d customers", len(df))
        return df

    # ── churn features ────────────────────────────────────────────────────────

    def churn_features(self) -> pd.DataFrame:
        """Features for churn probability scoring.

        Window-based features computed over 30/60/90-day lookback periods
        plus balance and transaction trends.
        """
        sql = """
            WITH customer_tx AS (
                SELECT
                    t.customer_sk,
                    t.amount,
                    t.balance_after,
                    t.transaction_type,
                    d.full_date,
                    d.date_key
                FROM fact_transactions t
                JOIN dim_date d ON t.date_key = d.date_key
            ),
            customer_stats AS (
                SELECT
                    c.customer_sk,
                    -- recency
                    EXTRACT(DAYS FROM AGE(CURRENT_DATE, MAX(ct.full_date))) AS days_since_last_transaction,
                    -- counts by window
                    COUNT(*) AS lifetime_tx_count,
                    SUM(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN 1 ELSE 0 END) AS tx_count_last_30d,
                    SUM(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '60 days' THEN 1 ELSE 0 END) AS tx_count_last_60d,
                    SUM(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '90 days' THEN 1 ELSE 0 END) AS tx_count_last_90d,
                    -- amount trends (last 30 vs prior 30)
                    SUM(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN ct.amount ELSE 0 END) AS amount_last_30d,
                    SUM(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '60 days'
                             AND ct.full_date < CURRENT_DATE - INTERVAL '30 days' THEN ct.amount ELSE 0 END) AS amount_prev_30d,
                    -- balance trend
                    AVG(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '30 days' THEN ct.balance_after END) AS avg_balance_last_30d,
                    AVG(CASE WHEN ct.full_date >= CURRENT_DATE - INTERVAL '60 days'
                             AND ct.full_date < CURRENT_DATE - INTERVAL '30 days' THEN ct.balance_after END) AS avg_balance_prev_30d,
                    -- product count
                    COUNT(DISTINCT t.product_sk) AS product_count
                FROM dim_customer c
                LEFT JOIN customer_tx ct ON c.customer_sk = ct.customer_sk
                LEFT JOIN fact_transactions t ON c.customer_sk = t.customer_sk
                WHERE c.is_current = TRUE
                GROUP BY c.customer_sk
            )
            SELECT
                cs.customer_sk,
                c.customer_id,
                c.income_bracket,
                c.customer_segment,
                EXTRACT(DAYS FROM AGE(CURRENT_DATE, c.account_open_date)) / 30.0 AS tenure_months,
                cs.days_since_last_transaction,
                cs.tx_count_last_30d,
                cs.tx_count_last_60d,
                cs.tx_count_last_90d,
                cs.lifetime_tx_count,
                cs.amount_last_30d,
                cs.amount_prev_30d,
                CASE
                    WHEN cs.amount_prev_30d > 0 THEN (cs.amount_last_30d - cs.amount_prev_30d) / cs.amount_prev_30d
                    ELSE 0
                END AS amount_trend,
                cs.avg_balance_last_30d,
                cs.avg_balance_prev_30d,
                CASE
                    WHEN cs.avg_balance_prev_30d > 0 THEN (cs.avg_balance_last_30d - cs.avg_balance_prev_30d) / cs.avg_balance_prev_30d
                    ELSE 0
                END AS balance_trend,
                cs.product_count,
                CASE WHEN cs.days_since_last_transaction >= 60 THEN 1 ELSE 0 END AS is_inactive
            FROM customer_stats cs
            JOIN dim_customer c ON cs.customer_sk = c.customer_sk
            WHERE c.is_current = TRUE
        """
        df = self._query_df(sql)
        if df.empty:
            logger.warning("churn_features: no data returned")
            return pd.DataFrame()

        df = df.set_index("customer_sk")
        numeric_cols = [
            "tenure_months", "days_since_last_transaction",
            "tx_count_last_30d", "tx_count_last_60d", "tx_count_last_90d",
            "lifetime_tx_count", "amount_last_30d", "amount_prev_30d",
            "amount_trend", "avg_balance_last_30d", "avg_balance_prev_30d",
            "balance_trend", "product_count", "is_inactive",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info("churn_features: %d customers", len(df))
        return df

    # ── cross-sell features ───────────────────────────────────────────────────

    def cross_sell_features(self) -> pd.DataFrame:
        """Features for cross-sell recommendation.

        Product ownership, co-occurrence affinity scores, demographic
        matrices, category preferences, and channel preferences.
        """
        df = self._query_df("""
            SELECT
                c.customer_sk,
                c.income_bracket,
                c.customer_segment,
                EXTRACT(YAGE FROM AGE(CURRENT_DATE, c.date_of_birth)) AS age,
                p.product_category,
                t.channel,
                t.transaction_category,
                t.amount
            FROM dim_customer c
            JOIN fact_transactions t ON c.customer_sk = t.customer_sk
            JOIN dim_product p ON t.product_sk = p.product_sk
            WHERE c.is_current = TRUE
        """)

        if df.empty:
            logger.warning("cross_sell_features: no data returned")
            return pd.DataFrame()

        df["age"] = pd.to_numeric(df["age"], errors="coerce")

        # ── current products per customer ──────────────────────────────────────
        product_sets = (
            df.groupby("customer_sk")["product_category"]
            .agg(lambda x: set(x))
            .rename("current_products")
        )

        # ── product counts ─────────────────────────────────────────────────────
        product_counts = (
            df.groupby(["customer_sk", "product_category"])["amount"]
            .count()
            .unstack(fill_value=0)
            .add_prefix("prod_count_")
        )

        # ── co-occurrence affinity (Jaccard-ish: what fraction of customers who own
        #    product A also own product B?) ──────────────────────────────────────
        product_dummy = (
            df[["customer_sk", "product_category"]]
            .drop_duplicates()
            .assign(has_product=1)
            .pivot(index="customer_sk", columns="product_category", values="has_product")
            .fillna(0)
        )

        n_customers = product_dummy.shape[0]
        affinity_scores: Dict[str, pd.Series] = {}
        for prod_a in product_dummy.columns:
            customers_a = set(product_dummy[product_dummy[prod_a] == 1].index)
            scores = {}
            for prod_b in product_dummy.columns:
                customers_b = set(product_dummy[product_dummy[prod_b] == 1].index)
                if len(customers_a) == 0:
                    scores[prod_b] = 0.0
                else:
                    scores[prod_b] = len(customers_a & customers_b) / len(customers_a)
            affinity_scores[prod_a] = pd.Series(scores)

        # avg affinity across owned products → higher = more "mainstream" bundle
        def avg_affinity(row: pd.Series) -> float:
            owned = [p for p in product_dummy.columns if row.get(f"count_{p}", 0) > 0]
            if not owned:
                return 0.0
            return np.mean([affinity_scores[p].get(name, 0) for p in owned])

        # ── demographic bins ───────────────────────────────────────────────────
        demo = (
            df[["customer_sk", "income_bracket", "age"]]
            .drop_duplicates()
            .set_index("customer_sk")
        )
        demo["age_group"] = pd.cut(
            demo["age"].fillna(0),
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
            right=False,
        )

        # ── category preferences ───────────────────────────────────────────────
        cat_pref = (
            df.groupby(["customer_sk", "transaction_category"])["amount"]
            .sum()
            .unstack(fill_value=0)
            .add_prefix("cat_spend_")
        )

        # ── channel preferences ────────────────────────────────────────────────
        chan_pref = (
            df.groupby(["customer_sk", "channel"])["amount"]
            .sum()
            .unstack(fill_value=0)
            .add_prefix("chan_spend_")
        )

        # ── assemble final DataFrame ───────────────────────────────────────────
        result = (
            product_sets.to_frame()
            .join(product_counts)
            .join(demo[["income_bracket", "age_group"]])
            .join(cat_pref)
            .join(chan_pref)
        )

        logger.info("cross_sell_features: %d customers", len(result))
        return result

    # ── monthly customer snapshot ─────────────────────────────────────────────

    def monthly_customer_snapshot(self, snapshot_date: str) -> pd.DataFrame:
        """Monthly snapshot of customer metrics for fact_customer_snapshot.

        Args:
            snapshot_date: ISO date string (YYYY-MM-DD) for the snapshot point.
        """
        snapshot_dt = pd.to_datetime(snapshot_date).date()
        snapshot_key = int(snapshot_dt.strftime("%Y%m%d"))

        sql = """
            SELECT
                :snapshot_key AS date_key,
                c.customer_sk,
                COUNT(t.transaction_sk) AS total_transactions,
                COALESCE(SUM(t.amount), 0) AS total_amount,
                COALESCE(AVG(t.amount), 0) AS avg_transaction_amount,
                MAX(t.balance_after) AS account_balance,
                COUNT(DISTINCT t.product_sk) AS num_products,
                EXTRACT(DAYS FROM AGE(:snapshot_date::date, MAX(d.full_date))) AS days_since_last_transaction,
                CASE
                    WHEN MAX(d.full_date) >= :snapshot_date::date - INTERVAL '30 days' THEN TRUE
                    ELSE FALSE
                END AS is_active
            FROM dim_customer c
            LEFT JOIN fact_transactions t
                ON c.customer_sk = t.customer_sk
                AND d.full_date <= :snapshot_date::date
            LEFT JOIN dim_date d ON t.date_key = d.date_key
            WHERE c.is_current = TRUE
            GROUP BY c.customer_sk
        """
        df = self._query_df(sql, {"snapshot_key": snapshot_key, "snapshot_date": snapshot_date})
        if df.empty:
            logger.warning("monthly_customer_snapshot: no data for %s", snapshot_date)
            return pd.DataFrame()

        df["date_key"] = df["date_key"].astype(int)
        for col in ["total_transactions", "total_amount", "avg_transaction_amount",
                     "account_balance", "num_products", "days_since_last_transaction"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df["is_active"] = df["is_active"].astype(bool)

        logger.info("monthly_customer_snapshot: %d customers for %s", len(df), snapshot_date)
        return df

    # ── product performance ───────────────────────────────────────────────────

    def product_performance_view(self) -> pd.DataFrame:
        """Product-level analytics: revenue, volume, customer count, avg balance."""
        sql = """
            SELECT
                p.product_sk,
                p.product_id,
                p.product_name,
                p.product_category,
                p.product_type,
                COUNT(t.transaction_sk) AS transaction_count,
                COUNT(DISTINCT t.customer_sk) AS customer_count,
                SUM(t.amount) AS total_revenue,
                AVG(t.amount) AS avg_transaction_amount,
                AVG(t.balance_after) AS avg_balance,
                MIN(d.full_date) AS first_transaction_date,
                MAX(d.full_date) AS last_transaction_date
            FROM dim_product p
            LEFT JOIN fact_transactions t ON p.product_sk = t.product_sk
            LEFT JOIN dim_date d ON t.date_key = d.date_key
            WHERE p.is_active = TRUE
            GROUP BY p.product_sk, p.product_id, p.product_name,
                     p.product_category, p.product_type
            ORDER BY total_revenue DESC NULLS LAST
        """
        df = self._query_df(sql)
        if df.empty:
            logger.warning("product_performance_view: no data returned")
            return pd.DataFrame()

        for col in ["transaction_count", "customer_count", "total_revenue",
                     "avg_transaction_amount", "avg_balance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info("product_performance_view: %d products", len(df))
        return df

    # ── regional analysis ─────────────────────────────────────────────────────

    def regional_analysis_view(self) -> pd.DataFrame:
        """Geographic analytics: transactions, revenue, customer count by state/city."""
        sql = """
            SELECT
                c.state,
                c.city,
                COUNT(t.transaction_sk) AS transaction_count,
                COUNT(DISTINCT t.customer_sk) AS customer_count,
                SUM(t.amount) AS total_revenue,
                AVG(t.amount) AS avg_transaction_amount,
                AVG(t.balance_after) AS avg_balance
            FROM fact_transactions t
            JOIN dim_customer c ON t.customer_sk = c.customer_sk
            WHERE c.is_current = TRUE
            GROUP BY c.state, c.city
            ORDER BY total_revenue DESC NULLS LAST
        """
        df = self._query_df(sql)
        if df.empty:
            logger.warning("regional_analysis_view: no data returned")
            return pd.DataFrame()

        for col in ["transaction_count", "customer_count", "total_revenue",
                     "avg_transaction_amount", "avg_balance"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info("regional_analysis_view: %d state/city groups", len(df))
        return df

    # ── fraud detection features ──────────────────────────────────────────────

    def fraud_detection_features(self) -> pd.DataFrame:
        """Features for fraud detection.

        Per-customer z-scores, recency, merchant risk, unusual-hour flag,
        and velocity (transactions per hour).
        """
        sql = """
            WITH customer_stats AS (
                SELECT
                    t.customer_sk,
                    AVG(t.amount) AS mean_amount,
                    STDDEV(t.amount) AS stddev_amount
                FROM fact_transactions t
                GROUP BY t.customer_sk
            ),
            merchant_risk AS (
                SELECT
                    t.merchant_category,
                    AVG(CASE WHEN t.is_fraudulent THEN 1.0 ELSE 0.0 END) AS fraud_rate
                FROM fact_transactions t
                WHERE t.merchant_category IS NOT NULL
                GROUP BY t.merchant_category
            ),
            hourly_velocity AS (
                SELECT
                    t.customer_sk,
                    EXTRACT(HOUR FROM t.created_at) AS tx_hour,
                    COUNT(*) AS tx_count,
                    COUNT(*) / NULLIF(EXTRACT(EPOCH FROM (MAX(t.created_at) - MIN(t.created_at))) / 3600, 0) AS tx_per_hour
                FROM fact_transactions t
                GROUP BY t.customer_sk, EXTRACT(HOUR FROM t.created_at)
            ),
            ranked_tx AS (
                SELECT
                    t.transaction_sk,
                    t.customer_sk,
                    t.amount,
                    t.merchant_category,
                    t.channel,
                    t.created_at,
                    EXTRACT(HOUR FROM t.created_at) AS tx_hour,
                    d.full_date,
                    LAG(t.created_at) OVER (PARTITION BY t.customer_sk ORDER BY t.created_at) AS prev_tx_time
                FROM fact_transactions t
                JOIN dim_date d ON t.date_key = d.date_key
            )
            SELECT
                r.transaction_sk,
                r.customer_sk,
                r.amount,
                r.merchant_category,
                r.channel,
                r.created_at,
                r.tx_hour,
                -- z-score within customer history
                CASE
                    WHEN cs.stddev_amount > 0 THEN (r.amount - cs.mean_amount) / cs.stddev_amount
                    ELSE 0
                END AS amount_zscore,
                -- time since last transaction (seconds)
                EXTRACT(EPOCH FROM (r.created_at - r.prev_tx_time)) AS time_since_last_tx,
                -- merchant category risk
                COALESCE(mr.fraud_rate, 0) AS merchant_risk_score,
                -- unusual hour flag (midnight–5am)
                CASE WHEN r.tx_hour < 6 THEN 1 ELSE 0 END AS unusual_hour_flag,
                -- velocity
                hv.tx_per_hour AS hourly_velocity
            FROM ranked_tx r
            LEFT JOIN customer_stats cs ON r.customer_sk = cs.customer_sk
            LEFT JOIN merchant_risk mr ON r.merchant_category = mr.merchant_category
            LEFT JOIN hourly_velocity hv ON r.customer_sk = hv.customer_sk AND r.tx_hour = hv.tx_hour
            ORDER BY r.customer_sk, r.created_at
        """
        df = self._query_df(sql)
        if df.empty:
            logger.warning("fraud_detection_features: no data returned")
            return pd.DataFrame()

        for col in ["amount_zscore", "time_since_last_tx", "merchant_risk_score",
                     "unusual_hour_flag", "hourly_velocity"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        logger.info("fraud_detection_features: %d transactions", len(df))
        return df

    # ── view creation helper ──────────────────────────────────────────────────

    def create_view_as_sql(self, view_name: str, query: str) -> str:
        """Generate CREATE VIEW SQL statement."""
        return f'CREATE OR REPLACE VIEW "{view_name}" AS\n{query}'

    # ── materialize all ───────────────────────────────────────────────────────

    def materialize_all_views(self) -> None:
        """Materialize all analytical views as tables."""
        views = [
            ("vw_customer_features", "customer_features_view"),
            ("vw_churn_features", "churn_features"),
            ("vw_cross_sell_features", "cross_sell_features"),
            ("vw_product_performance", "product_performance_view"),
            ("vw_regional_analysis", "regional_analysis_view"),
            ("vw_fraud_detection_features", "fraud_detection_features"),
        ]

        with self.engine.begin() as conn:
            for table_name, method_name in views:
                logger.info("Materializing %s from %s()", table_name, method_name)
                method = getattr(self, method_name)
                df = method()
                if df.empty:
                    logger.warning("Skipping %s — source returned 0 rows", table_name)
                    continue
                df.to_sql(
                    table_name,
                    con=conn,
                    if_exists="replace",
                    index=True,
                    chunksize=5000,
                )
                logger.info("Wrote %d rows to %s", len(df), table_name)

        logger.info("materialize_all_views complete")
