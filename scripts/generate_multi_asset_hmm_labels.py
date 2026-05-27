#!/usr/bin/env python3
"""
Generate HMM regime labels for multiple ETFs and write a combined long CSV.

The per-asset labeling logic intentionally reuses hmm_regime_labeling.py so the
multi-asset output stays comparable with the existing SPY labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hmm_regime_labeling import (
    LABEL_TO_CODE,
    download_stooq,
    download_yahoo_chart,
    load_market_csv,
    make_features,
    run_rolling_hmm,
    write_labeled_csv,
)


def asset_key(symbol: str) -> str:
    key = symbol.strip().upper()
    if key.endswith(".US"):
        key = key[:-3]
    if not key:
        raise ValueError("Asset symbol cannot be empty")
    return key


def raw_path_for(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / f"{asset_key(symbol).lower()}_daily.csv"


def label_path_for(processed_dir: Path, symbol: str, fit_step: int) -> Path:
    return processed_dir / f"{asset_key(symbol).lower()}_hmm_regime_labels_{fit_step}d.csv"


def download_asset(symbol: str, start: str, end: str, raw_path: Path) -> None:
    print(f"Downloading {symbol} to {raw_path}", file=sys.stderr)
    try:
        download_yahoo_chart(symbol, start, end, raw_path)
        return
    except Exception as yahoo_error:
        print(f"Yahoo download failed for {symbol}: {yahoo_error}", file=sys.stderr)

    stooq_symbols = [symbol]
    if "." not in symbol:
        stooq_symbols.append(f"{symbol}.US")
    last_error: Optional[Exception] = None
    for stooq_symbol in stooq_symbols:
        try:
            download_stooq(stooq_symbol, start, end, raw_path)
            return
        except Exception as stooq_error:
            last_error = stooq_error
            print(f"Stooq download failed for {stooq_symbol}: {stooq_error}", file=sys.stderr)
    raise RuntimeError(f"Could not download {symbol}") from last_error


def generate_labels_for_asset(args: argparse.Namespace, symbol: str) -> Path:
    raw_path = raw_path_for(args.raw_dir, symbol)
    out_path = label_path_for(args.processed_dir, symbol, args.fit_step)

    if out_path.exists() and not args.refresh_labels and not args.refresh_raw:
        print(f"Using existing labels for {symbol}: {out_path}", file=sys.stderr)
        return out_path

    if args.refresh_raw or not raw_path.exists():
        download_asset(symbol, args.start, args.end, raw_path)
    else:
        print(f"Using existing raw data for {symbol}: {raw_path}", file=sys.stderr)

    market = load_market_csv(raw_path)
    if len(market.dates) < args.train_window + args.drawdown_window:
        raise ValueError(
            f"{symbol} has too few rows: {len(market.dates)}. "
            f"Need more than train_window + warmup windows."
        )

    features = make_features(
        market,
        vol_window=args.vol_window,
        fast_ma=args.fast_ma,
        slow_ma=args.slow_ma,
        drawdown_window=args.drawdown_window,
    )
    points = run_rolling_hmm(
        features,
        train_window=args.train_window,
        fit_step=args.fit_step,
        n_states=args.states,
        max_iter=args.max_iter,
        n_init=args.n_init,
        seed=args.seed,
    )
    write_labeled_csv(
        out_path,
        market,
        features,
        points,
        target_horizon=args.target_horizon,
        smoothing_window=args.smoothing_window,
    )

    counts: Dict[str, int] = {label: 0 for label in LABEL_TO_CODE}
    for point in points:
        counts[point.label] += 1
    print(f"Wrote {len(points)} {symbol} labels to {out_path}", file=sys.stderr)
    print(f"{symbol} label counts: {counts}", file=sys.stderr)
    return out_path


def combine_label_files(label_files: Dict[str, Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Optional[csv.DictWriter] = None
    fieldnames: Optional[List[str]] = None

    with out_path.open("w", encoding="utf-8", newline="") as out_file:
        for symbol, label_file in label_files.items():
            with label_file.open("r", encoding="utf-8-sig", newline="") as in_file:
                reader = csv.DictReader(in_file)
                if reader.fieldnames is None:
                    raise ValueError(f"No header found in {label_file}")
                current_fields = ["asset"] + list(reader.fieldnames)
                if fieldnames is None:
                    fieldnames = current_fields
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise ValueError(
                        f"Schema mismatch in {label_file}. "
                        f"Expected {fieldnames}, got {current_fields}"
                    )
                assert writer is not None
                for row in reader:
                    row_with_asset = {"asset": asset_key(symbol)}
                    row_with_asset.update(row)
                    writer.writerow(row_with_asset)

    print(f"Wrote combined labels to {out_path}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate multi-asset HMM regime labels")
    parser.add_argument("--assets", nargs="+", default=["SPY", "QQQ", "GLD", "TLT"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-05-15")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
    )
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--refresh-labels", action="store_true")
    parser.add_argument("--train-window", type=int, default=504)
    parser.add_argument("--fit-step", type=int, default=5)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--n-init", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--fast-ma", type=int, default=20)
    parser.add_argument("--slow-ma", type=int, default=60)
    parser.add_argument("--drawdown-window", type=int, default=126)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--target-horizon", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.combined_output is None:
        args.combined_output = args.processed_dir / f"multi_asset_hmm_regime_labels_{args.fit_step}d.csv"
    label_files: Dict[str, Path] = {}
    for symbol in args.assets:
        key = asset_key(symbol)
        label_files[key] = generate_labels_for_asset(args, symbol)
    combine_label_files(label_files, args.combined_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
