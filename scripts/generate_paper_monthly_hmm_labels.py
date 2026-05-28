#!/usr/bin/env python3
"""
Generate paper-style monthly HMM labels for the ETF universes in the reference paper.

Paper-aligned choices:
  - Yahoo Finance adjusted prices
  - monthly returns
  - 3 hidden states
  - 24-month sliding training window
  - 1-month rebalance/label step
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from generate_multi_asset_hmm_labels import asset_key, download_asset
from hmm_regime_labeling import (
    CODE_TO_LABEL,
    LABEL_TO_CODE,
    HMMParams,
    fit_gaussian_hmm,
    forward_backward,
    load_market_csv,
    log_gaussian_diag,
    parse_date,
    viterbi,
)


ASSET10 = ["SPY", "IEV", "EWJ", "EEM", "TLT", "IEF", "IYR", "RWX", "GLD", "DBC"]
ASSET22 = [
    "JKD",
    "IJR",
    "IWM",
    "IEV",
    "EWJ",
    "EWY",
    "EFA",
    "EEM",
    "TLT",
    "IEF",
    "TIP",
    "AGG",
    "EMB",
    "GTIP",
    "HYT",
    "IYR",
    "RWX",
    "OIL",
    "GLD",
    "UUP",
    "DBC",
    "CPER",
]
PAPER23 = list(dict.fromkeys(ASSET10 + ASSET22))


@dataclass
class MonthlyPoint:
    date: str
    close: float
    monthly_return: float


@dataclass
class MonthlyLabel:
    date: str
    close: float
    monthly_return: float
    raw_state: int
    hmm_label: str
    hmm_label_code: int
    prob_bear: float
    prob_neutral: float
    prob_bull: float
    state_mean_monthly_return: float
    state_ann_return: float
    state_ann_vol: float
    state_sharpe: float
    state_count: int
    paper_phase: str
    paper_selected: int
    model_loglik: float


def raw_daily_path(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / "paper_etfs_daily" / f"{asset_key(symbol).lower()}_daily.csv"


def raw_monthly_path(raw_dir: Path, symbol: str) -> Path:
    return raw_dir / "paper_etfs_monthly" / f"{asset_key(symbol).lower()}_monthly.csv"


def label_path(processed_dir: Path, symbol: str) -> Path:
    return processed_dir / f"{asset_key(symbol).lower()}_paper_monthly_hmm_labels.csv"


def monthly_points_from_daily(daily_csv: Path) -> List[MonthlyPoint]:
    market = load_market_csv(daily_csv)
    month_last: Dict[Tuple[int, int], Tuple[str, float]] = {}
    for d, close in zip(market.dates, market.close):
        month_last[(d.year, d.month)] = (d.isoformat(), float(close))

    rows = [month_last[key] for key in sorted(month_last)]
    points: List[MonthlyPoint] = []
    previous_close: Optional[float] = None
    for date_string, close in rows:
        if previous_close is None:
            previous_close = close
            continue
        monthly_return = close / previous_close - 1.0
        points.append(MonthlyPoint(date=date_string, close=close, monthly_return=monthly_return))
        previous_close = close
    return points


def write_monthly_csv(path: Path, points: List[MonthlyPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "close", "monthly_return"])
        writer.writeheader()
        for point in points:
            writer.writerow(asdict(point))


def annualized_return(mean_monthly_return: float) -> float:
    return (1.0 + mean_monthly_return) ** 12.0 - 1.0 if mean_monthly_return > -1.0 else -1.0


def state_stats(states: np.ndarray, returns: np.ndarray, n_states: int) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    fallback_vol = float(np.std(returns, ddof=1) * math.sqrt(12.0)) if len(returns) > 1 else 0.0
    for state in range(n_states):
        state_returns = returns[states == state]
        count = int(len(state_returns))
        if count == 0:
            mean_monthly = -np.inf
            ann_return = -np.inf
            ann_vol = fallback_vol
            sharpe = -np.inf
        else:
            mean_monthly = float(np.mean(state_returns))
            ann_return = annualized_return(mean_monthly)
            ann_vol = (
                float(np.std(state_returns, ddof=1) * math.sqrt(12.0))
                if count > 1
                else fallback_vol
            )
            sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.sign(ann_return) * np.inf
        stats[state] = {
            "count": count,
            "mean_monthly_return": mean_monthly,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        }
    return stats


def state_label_mapping(stats: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    states = list(stats)
    bull_state = max(states, key=lambda s: (stats[s]["sharpe"], stats[s]["ann_return"]))
    bear_state = min(states, key=lambda s: (stats[s]["sharpe"], stats[s]["ann_return"]))
    return {
        state: "Bull" if state == bull_state else "Bear" if state == bear_state else "Neutral"
        for state in states
    }


def label_monthly_points(
    points: List[MonthlyPoint],
    train_window: int,
    n_states: int,
    max_iter: int,
    n_init: int,
    seed: int,
) -> List[MonthlyLabel]:
    if len(points) < train_window:
        raise ValueError(f"Need at least {train_window} monthly returns, got {len(points)}")

    returns = np.array([p.monthly_return for p in points], dtype=float)
    labels: List[MonthlyLabel] = []

    for end_idx in range(train_window - 1, len(points)):
        window_returns = returns[end_idx - train_window + 1 : end_idx + 1]
        x = window_returns.reshape(-1, 1)
        params: HMMParams = fit_gaussian_hmm(
            x,
            n_states=n_states,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed + end_idx,
        )
        log_b = log_gaussian_diag(x, params.means, params.covars)
        alpha, beta, loglik = forward_backward(log_b, params.pi, params.trans)
        gamma = np.exp(alpha + beta - loglik)
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        states = viterbi(log_b, params.pi, params.trans)
        stats = state_stats(states, window_returns, n_states)
        mapping = state_label_mapping(stats)

        current_state = int(states[-1])
        label = mapping[current_state]
        probs_by_label = {name: 0.0 for name in LABEL_TO_CODE}
        for raw_state, label_name in mapping.items():
            probs_by_label[label_name] += float(gamma[-1, raw_state])

        best_state = max(stats, key=lambda s: (stats[s]["sharpe"], stats[s]["ann_return"]))
        current_stats = stats[current_state]
        paper_selected = int(current_state == best_state and current_stats["sharpe"] > 0.0)
        paper_phase = "Increasing" if current_stats["sharpe"] > 0.0 else "Decreasing"
        point = points[end_idx]
        labels.append(
            MonthlyLabel(
                date=point.date,
                close=point.close,
                monthly_return=point.monthly_return,
                raw_state=current_state,
                hmm_label=label,
                hmm_label_code=LABEL_TO_CODE[label],
                prob_bear=probs_by_label["Bear"],
                prob_neutral=probs_by_label["Neutral"],
                prob_bull=probs_by_label["Bull"],
                state_mean_monthly_return=float(current_stats["mean_monthly_return"]),
                state_ann_return=float(current_stats["ann_return"]),
                state_ann_vol=float(current_stats["ann_vol"]),
                state_sharpe=float(current_stats["sharpe"]),
                state_count=int(current_stats["count"]),
                paper_phase=paper_phase,
                paper_selected=paper_selected,
                model_loglik=float(loglik),
            )
        )
    return labels


def write_labels_csv(path: Path, labels: List[MonthlyLabel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(labels[0]).keys()))
        writer.writeheader()
        for label in labels:
            writer.writerow(asdict(label))


def combine_labels(symbols: Sequence[str], processed_dir: Path, output: Path) -> int:
    fieldnames: Optional[List[str]] = None
    rows_written = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as out_file:
        writer: Optional[csv.DictWriter] = None
        for symbol in symbols:
            path = label_path(processed_dir, symbol)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8-sig", newline="") as in_file:
                reader = csv.DictReader(in_file)
                if reader.fieldnames is None:
                    continue
                current_fields = ["asset"] + list(reader.fieldnames)
                if fieldnames is None:
                    fieldnames = current_fields
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                    writer.writeheader()
                elif current_fields != fieldnames:
                    raise ValueError(f"Schema mismatch in {path}")
                assert writer is not None
                for row in reader:
                    out = {"asset": asset_key(symbol)}
                    out.update(row)
                    writer.writerow(out)
                    rows_written += 1
    return rows_written


def process_asset(args: argparse.Namespace, symbol: str) -> Dict[str, object]:
    key = asset_key(symbol)
    daily_path = raw_daily_path(args.raw_dir, key)
    monthly_path = raw_monthly_path(args.raw_dir, key)
    labels_path = label_path(args.processed_dir, key)

    if args.refresh_raw or not daily_path.exists():
        print(f"Downloading {key}", file=sys.stderr)
        download_asset(key, args.start, args.end, daily_path)

    points = monthly_points_from_daily(daily_path)
    if args.start:
        start_date = parse_date(args.start)
        points = [p for p in points if parse_date(p.date) >= start_date]
    if args.end:
        end_date = parse_date(args.end)
        points = [p for p in points if parse_date(p.date) <= end_date]
    write_monthly_csv(monthly_path, points)

    labels = label_monthly_points(
        points,
        train_window=args.train_window,
        n_states=args.states,
        max_iter=args.max_iter,
        n_init=args.n_init,
        seed=args.seed,
    )
    write_labels_csv(labels_path, labels)
    counts = {name: 0 for name in LABEL_TO_CODE}
    for label in labels:
        counts[CODE_TO_LABEL[label.hmm_label_code]] += 1
    print(f"Wrote {len(labels)} monthly labels for {key}: {counts}", file=sys.stderr)
    return {
        "asset": key,
        "daily_raw": str(daily_path),
        "monthly_raw": str(monthly_path),
        "labels": str(labels_path),
        "monthly_rows": len(points),
        "label_rows": len(labels),
        "first_month": points[0].date if points else None,
        "last_month": points[-1].date if points else None,
        "label_counts": counts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate paper-style monthly HMM labels")
    parser.add_argument("--assets", nargs="+", default=PAPER23)
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default="2026-05-27")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--train-window", type=int, default=24)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combined-output", type=Path, default=Path("data/processed/paper23_monthly_hmm_labels.csv"))
    parser.add_argument("--asset10-output", type=Path, default=Path("data/processed/paper_asset10_monthly_hmm_labels.csv"))
    parser.add_argument("--asset22-output", type=Path, default=Path("data/processed/paper_asset22_monthly_hmm_labels.csv"))
    parser.add_argument("--meta-output", type=Path, default=Path("data/processed/paper23_monthly_hmm_labels_meta.json"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    symbols = [asset_key(symbol) for symbol in args.assets]

    summaries: List[Dict[str, object]] = []
    failures: Dict[str, str] = {}
    for symbol in symbols:
        try:
            summaries.append(process_asset(args, symbol))
        except Exception as exc:
            failures[symbol] = str(exc)
            print(f"Failed {symbol}: {exc}", file=sys.stderr)
            if args.strict:
                raise

    available = [str(summary["asset"]) for summary in summaries]
    combined_rows = combine_labels(available, args.processed_dir, args.combined_output)
    asset10_rows = combine_labels([s for s in ASSET10 if s in available], args.processed_dir, args.asset10_output)
    asset22_rows = combine_labels([s for s in ASSET22 if s in available], args.processed_dir, args.asset22_output)

    meta = {
        "method": "paper-style monthly HMM labels: Yahoo adjusted close, monthly returns, 24-month sliding window, 1-month step, 3 hidden states",
        "requested_assets": symbols,
        "available_assets": available,
        "failed_assets": failures,
        "start": args.start,
        "end": args.end,
        "train_window_months": args.train_window,
        "states": args.states,
        "outputs": {
            "paper23": str(args.combined_output),
            "asset10": str(args.asset10_output),
            "asset22": str(args.asset22_output),
            "meta": str(args.meta_output),
        },
        "rows": {
            "paper23": combined_rows,
            "asset10": asset10_rows,
            "asset22": asset22_rows,
        },
        "assets": summaries,
        "label_encoding": LABEL_TO_CODE,
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return 1 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
