#!/usr/bin/env python3
"""CLI entry point for the Bank Data Warehouse pipeline.

Usage
-----
    python run.py full          # run every stage end-to-end
    python run.py generate      # synthetic data generation only
    python run.py load          # ETL into warehouse only
    python run.py quality       # data-quality checks on existing tables
    python run.py views         # rebuild analytical views
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure package is importable when invoked from the repo root.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_warehouse.src.pipeline import (
    _load_config,
    _get_engine,
    create_tables,
    generate_data,
    load_to_warehouse,
    run_quality_checks,
    apply_scd2,
    create_analytical_views,
    save_outputs,
)

logger = logging.getLogger("data_warehouse")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---- Sub-commands ----------------------------------------------------------

def cmd_full(args: argparse.Namespace) -> None:
    """Run the entire pipeline."""
    from data_warehouse.src.pipeline import run_pipeline
    run_pipeline(config_path=args.config)


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic data and load into an empty warehouse."""
    cfg = _load_config(args.config)
    engine = _get_engine(cfg)
    create_tables(engine)
    data = generate_data(cfg)
    load_to_warehouse(engine, data)
    logger.info("Data generation and load complete.")


def cmd_load(args: argparse.Namespace) -> None:
    """Regenerate source data and ETL-load into existing tables."""
    cfg = _load_config(args.config)
    engine = _get_engine(cfg)
    data = generate_data(cfg)
    load_to_warehouse(engine, data)
    logger.info("ETL load complete.")


def cmd_quality(args: argparse.Namespace) -> None:
    """Run data-quality checks on tables already in the warehouse."""
    cfg = _load_config(args.config)
    engine = _get_engine(cfg)
    failures = run_quality_checks(engine, cfg)
    if failures:
        for table, msgs in failures.items():
            for msg in msgs:
                logger.error("  %s: %s", table, msg)
        sys.exit(1)
    else:
        logger.info("All quality checks passed.")


def cmd_views(args: argparse.Namespace) -> None:
    """Rebuild analytical views and export to CSV."""
    cfg = _load_config(args.config)
    engine = _get_engine(cfg)
    create_analytical_views(engine, cfg)
    apply_scd2(engine, cfg)
    save_outputs(engine)
    logger.info("Analytical views rebuilt and exported.")


# ---- Argument parser -------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bank-dw",
        description="Bank Transaction Data Warehouse — CLI",
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        help="Path to a YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    sub = parser.add_subparsers(dest="command", help="Pipeline stage to run")
    sub.required = True

    sub.add_parser("full", help="Run the entire pipeline end-to-end")
    sub.add_parser("generate", help="Generate synthetic data and load into warehouse")
    sub.add_parser("load", help="Regenerate source data and ETL-load")
    sub.add_parser("quality", help="Run data-quality checks")
    sub.add_parser("views", help="Rebuild analytical views and export CSVs")

    return parser


DISPATCH = {
    "full": cmd_full,
    "generate": cmd_generate,
    "load": cmd_load,
    "quality": cmd_quality,
    "views": cmd_views,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    logger.info("Command: %s", args.command)
    try:
        DISPATCH[args.command](args)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
