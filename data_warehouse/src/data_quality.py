"""Great-Expectations-style data quality checks without requiring Great Expectations.

Provides a lightweight ``DataQualityChecker`` that validates DataFrames against
a declarative configuration and produces structured reports.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result model
# ------------------------------------------------------------------

@dataclass
class CheckResult:
    """Single check outcome."""

    check_type: str
    columns: List[str]
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# Checker
# ------------------------------------------------------------------

class DataQualityChecker:
    """Data quality checks for source data before ETL.

    Parameters
    ----------
    config : dict
        Global configuration (unused currently, reserved for future use).
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config: dict = config or {}
        self.results: List[CheckResult] = []
        logger.debug("DataQualityChecker initialised.")

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_not_null(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check that specified columns contain no null values.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        columns : list[str]
            Column names that must be non-null.

        Returns
        -------
        bool
            ``True`` if all checks pass, ``False`` otherwise.
        """
        passed = True
        for col in columns:
            if col not in df.columns:
                self._record("not_null", [col], False,
                             f"Column '{col}' does not exist in DataFrame.")
                passed = False
                continue
            null_count = int(df[col].isnull().sum())
            ok = null_count == 0
            self._record(
                "not_null", [col], ok,
                f"Column '{col}': {null_count} null(s) found.",
                {"null_count": null_count},
            )
            if not ok:
                passed = False
        return passed

    def check_unique(self, df: pd.DataFrame, columns: List[str]) -> bool:
        """Check that specified columns have unique values (no duplicates).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        columns : list[str]
            Column names that must be unique.

        Returns
        -------
        bool
            ``True`` if all checks pass.
        """
        passed = True
        for col in columns:
            if col not in df.columns:
                self._record("unique", [col], False,
                             f"Column '{col}' does not exist.")
                passed = False
                continue
            dup_count = int(df[col].duplicated().sum())
            ok = dup_count == 0
            self._record(
                "unique", [col], ok,
                f"Column '{col}': {dup_count} duplicate(s) found.",
                {"duplicate_count": dup_count},
            )
            if not ok:
                passed = False
        return passed

    def check_values_in_set(self, df: pd.DataFrame, column: str,
                            valid_values: List[Any]) -> bool:
        """Check that all values in a column belong to a valid set.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        column : str
            Column to check.
        valid_values : list
            Allowed values.

        Returns
        -------
        bool
            ``True`` if all values are in the set.
        """
        if column not in df.columns:
            self._record("values_in_set", [column], False,
                         f"Column '{column}' does not exist.")
            return False
        invalid = set(df[column].dropna().unique()) - set(valid_values)
        ok = len(invalid) == 0
        self._record(
            "values_in_set", [column], ok,
            f"Column '{column}': {len(invalid)} invalid value(s).",
            {"invalid_values": sorted(invalid)},
        )
        return ok

    def check_range(self, df: pd.DataFrame, column: str,
                    min_val: Optional[float] = None,
                    max_val: Optional[float] = None) -> bool:
        """Check that numeric column values fall within a range.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        column : str
            Column to check (must be numeric).
        min_val : float, optional
            Minimum allowed value (inclusive).
        max_val : float, optional
            Maximum allowed value (inclusive).

        Returns
        -------
        bool
            ``True`` if all values are within range.
        """
        if column not in df.columns:
            self._record("range", [column], False,
                         f"Column '{col}' does not exist.")
            return False
        series = df[column].dropna()
        violations: List[str] = []
        if min_val is not None:
            below = int((series < min_val).sum())
            if below:
                violations.append(f"{below} value(s) below {min_val}")
        if max_val is not None:
            above = int((series > max_val).sum())
            if above:
                violations.append(f"{above} value(s) above {max_val}")
        ok = len(violations) == 0
        self._record(
            "range", [column], ok,
            f"Column '{column}': {'; '.join(violations) if violations else 'all values in range.'}",
            {"min": min_val, "max": max_val},
        )
        return ok

    def check_regex(self, df: pd.DataFrame, column: str, pattern: str) -> bool:
        """Check that all non-null values match a regex pattern.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        column : str
            Column to check.
        pattern : str
            Regular expression pattern.

        Returns
        -------
        bool
            ``True`` if every value matches.
        """
        if column not in df.columns:
            self._record("regex", [column], False,
                         f"Column '{column}' does not exist.")
            return False
        compiled = re.compile(pattern)
        series = df[column].dropna().astype(str)
        mismatch = int((~series.str.match(compiled)).sum())
        ok = mismatch == 0
        self._record(
            "regex", [column], ok,
            f"Column '{column}': {mismatch} value(s) did not match pattern '{pattern}'.",
            {"pattern": pattern, "mismatch_count": mismatch},
        )
        return ok

    def check_date_format(self, df: pd.DataFrame, column: str,
                          fmt: str = "%Y-%m-%d") -> bool:
        """Check that a date column can be parsed with the given format.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        column : str
            Column to check.
        fmt : str
            Expected strptime format string.

        Returns
        -------
        bool
            ``True`` if all non-null values parse correctly.
        """
        if column not in df.columns:
            self._record("date_format", [column], False,
                         f"Column '{column}' does not exist.")
            return False
        series = df[column].dropna().astype(str)
        parse_errors = 0
        for val in series:
            try:
                datetime.strptime(val, fmt)
            except ValueError:
                parse_errors += 1
        ok = parse_errors == 0
        self._record(
            "date_format", [column], ok,
            f"Column '{column}': {parse_errors} value(s) failed to parse with format '{fmt}'.",
            {"format": fmt, "parse_errors": parse_errors},
        )
        return ok

    def check_referential_integrity(
        self,
        df: pd.DataFrame,
        column: str,
        reference_df: pd.DataFrame,
        ref_column: str,
    ) -> bool:
        """Check that every value in ``df[column]`` exists in ``reference_df[ref_column]``.

        Parameters
        ----------
        df : pd.DataFrame
            Child DataFrame.
        column : str
            Foreign-key column in *df*.
        reference_df : pd.DataFrame
            Parent (reference) DataFrame.
        ref_column : str
            Primary-key column in *reference_df*.

        Returns
        -------
        bool
            ``True`` if referential integrity holds.
        """
        if column not in df.columns:
            self._record("referential_integrity", [column], False,
                         f"Column '{column}' does not exist in child DataFrame.")
            return False
        if ref_column not in reference_df.columns:
            self._record("referential_integrity", [column], False,
                         f"Reference column '{ref_column}' does not exist in parent DataFrame.")
            return False
        child_vals = set(df[column].dropna().unique())
        parent_vals = set(reference_df[ref_column].dropna().unique())
        orphaned = child_vals - parent_vals
        ok = len(orphaned) == 0
        self._record(
            "referential_integrity", [column], ok,
            f"Column '{column}': {len(orphaned)} orphaned value(s).",
            {"orphaned_values": sorted(orphaned)[:20]},
        )
        return ok

    def check_freshness(self, df: pd.DataFrame, date_column: str,
                        max_age_days: int) -> bool:
        """Check that data is not older than *max_age_days* from today.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        date_column : str
            Column containing dates/timestamps.
        max_age_days : int
            Maximum acceptable age in days.

        Returns
        -------
        bool
            ``True`` if the most recent date is within the freshness window.
        """
        if date_column not in df.columns:
            self._record("freshness", [date_column], False,
                         f"Column '{date_column}' does not exist.")
            return False
        dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
        if dates.empty:
            self._record("freshness", [date_column], False,
                         f"Column '{date_column}' has no parseable dates.")
            return False
        most_recent = dates.max()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=max_age_days)
        ok = most_recent >= cutoff
        age_days = (pd.Timestamp.now() - most_recent).days
        self._record(
            "freshness", [date_column], ok,
            f"Column '{date_column}': most recent date is {age_days} day(s) old "
            f"(max allowed: {max_age_days}).",
            {"most_recent": str(most_recent), "age_days": age_days,
             "max_age_days": max_age_days},
        )
        return ok

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run_all_checks(self, df: pd.DataFrame, checks_config: dict) -> Dict[str, Any]:
        """Run all configured checks and return a structured report.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.
        checks_config : dict
            Mapping of check-type names to kwargs.  Example::

                {
                    "not_null": {"columns": ["customer_id", "email"]},
                    "unique":   {"columns": ["customer_id"]},
                    "range":    {"column": "amount", "min_val": 0},
                    "values_in_set": {
                        "column": "risk_category",
                        "valid_values": ["low", "medium", "high", "very_high"],
                    },
                    "regex":    {"column": "zip_code", "pattern": r"^\\d{5}$"},
                    "date_format": {"column": "transaction_date"},
                    "referential_integrity": {
                        "column": "customer_id",
                        "reference_df": customers_df,
                        "ref_column": "customer_id",
                    },
                    "freshness": {"date_column": "transaction_date", "max_age_days": 30},
                }

        Returns
        -------
        dict
            ``{"total": int, "passed": int, "failed": int, "results": [CheckResult, …]}``
        """
        dispatch: Dict[str, Callable[..., bool]] = {
            "not_null": lambda **kw: self.check_not_null(df, **kw),
            "unique": lambda **kw: self.check_unique(df, **kw),
            "values_in_set": lambda **kw: self.check_values_in_set(df, **kw),
            "range": lambda **kw: self.check_range(df, **kw),
            "regex": lambda **kw: self.check_regex(df, **kw),
            "date_format": lambda **kw: self.check_date_format(df, **kw),
            "referential_integrity": lambda **kw: self.check_referential_integrity(df, **kw),
            "freshness": lambda **kw: self.check_freshness(df, **kw),
        }

        for check_name, kwargs in checks_config.items():
            if check_name not in dispatch:
                logger.warning("Unknown check type '%s' — skipping.", check_name)
                continue
            try:
                dispatch[check_name](**kwargs)
            except Exception:
                logger.exception("Check '%s' raised an exception.", check_name)
                self._record(check_name, [], False, f"Check raised an exception.")

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        report: Dict[str, Any] = {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "results": self.results,
        }
        logger.info(
            "Checks complete: %d total, %d passed, %d failed.",
            len(self.results), passed, failed,
        )
        return report

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Generate a human-readable quality report from stored results.

        Returns
        -------
        str
            Multi-line report string.
        """
        if not self.results:
            return "No checks have been run."

        lines: List[str] = ["=" * 60, "  DATA QUALITY REPORT", "=" * 60, ""]
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        for i, r in enumerate(self.results, 1):
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] Check #{i}: {r.check_type}")
            lines.append(f"  Columns : {', '.join(r.columns) if r.columns else '(n/a)'}")
            lines.append(f"  Message : {r.message}")
            if r.details:
                lines.append(f"  Details : {r.details}")
            lines.append("")

        lines.append("-" * 60)
        lines.append(f"  Total: {len(self.results)}  |  Passed: {passed}  |  Failed: {failed}")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, check_type: str, columns: List[str], passed: bool,
                message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Append a ``CheckResult`` to the internal results list."""
        result = CheckResult(
            check_type=check_type,
            columns=columns,
            passed=passed,
            message=message,
            details=details or {},
        )
        self.results.append(result)
        log_fn = logger.info if passed else logger.warning
        log_fn("[%s] %s", check_type.upper(), message)

    def clear(self) -> None:
        """Reset all stored results."""
        self.results.clear()
        logger.debug("DataQualityChecker results cleared.")
