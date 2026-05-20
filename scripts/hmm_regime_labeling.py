#!/usr/bin/env python3
"""
Rolling Gaussian HMM regime labeling for a market index or ETF.

This script intentionally avoids pandas / sklearn / hmmlearn so it can run in a
minimal Python environment with only numpy installed.

Default project setting:
    SPY daily data -> rolling 504-day 3-state HMM -> Bull/Neutral/Bear labels
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


LABEL_TO_CODE = {"Bear": 0, "Neutral": 1, "Bull": 2}
CODE_TO_LABEL = {v: k for k, v in LABEL_TO_CODE.items()}


@dataclass
class MarketData:
    dates: List[date]
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray


@dataclass
class FeatureData:
    names: List[str]
    values: np.ndarray
    log_return: np.ndarray
    return_1d: np.ndarray
    vol_ann: np.ndarray
    ma_gap: np.ndarray
    drawdown: np.ndarray


@dataclass
class HMMParams:
    pi: np.ndarray
    trans: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    loglik: float


@dataclass
class RegimePoint:
    idx: int
    raw_state: int
    label: str
    prob_bear: float
    prob_neutral: float
    prob_bull: float
    state_mean_daily_return: float
    state_ann_return: float
    state_ann_vol: float
    state_sharpe: float
    state_count: int
    model_loglik: float


def parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unsupported date format: {value!r}")


def parse_float(value: Optional[str], default: float = np.nan) -> float:
    if value is None:
        return default
    value = value.strip().replace(",", "")
    if value == "" or value.lower() in {"nan", "null", "none"}:
        return default
    return float(value)


def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def download_stooq(symbol: str, start: str, end: str, out_path: Path) -> None:
    query_symbol = symbol.lower()
    d1 = start.replace("-", "")
    d2 = end.replace("-", "")
    url = f"https://stooq.com/q/d/l/?s={query_symbol}&i=d&d1={d1}&d2={d2}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
    text = content.decode("utf-8", errors="replace")
    if "No data" in text or "Get your apikey" in text or len(text.splitlines()) < 5:
        raise RuntimeError(f"Downloaded data appears empty for {symbol}: {url}")
    out_path.write_text(text, encoding="utf-8")


def download_yahoo_chart(symbol: str, start: str, end: str, out_path: Path) -> None:
    query_symbol = symbol.upper()
    if query_symbol.endswith(".US"):
        query_symbol = query_symbol[:-3]

    start_dt = datetime.combine(parse_date(start), datetime.min.time(), tzinfo=timezone.utc)
    # Yahoo's period2 is exclusive, so add one day to include the requested end.
    end_dt = datetime.combine(parse_date(end) + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{query_symbol}"
        f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true"
    )
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo Finance error for {query_symbol}: {chart['error']}")
    result = chart.get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo Finance returned no chart result for {query_symbol}")

    data = result[0]
    timestamps = data.get("timestamp") or []
    quote = (data.get("indicators", {}).get("quote") or [{}])[0]
    adjclose = (data.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose") or []

    rows: List[List[object]] = []
    for i, ts in enumerate(timestamps):
        o = quote.get("open", [None] * len(timestamps))[i]
        h = quote.get("high", [None] * len(timestamps))[i]
        l = quote.get("low", [None] * len(timestamps))[i]
        c = quote.get("close", [None] * len(timestamps))[i]
        v = quote.get("volume", [None] * len(timestamps))[i]
        ac = adjclose[i] if i < len(adjclose) else c
        if any(value is None for value in (o, h, l, c, ac)):
            continue
        day = datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        rows.append([day, o, h, l, c, ac, v or 0])

    if len(rows) < 50:
        raise RuntimeError(f"Yahoo Finance returned too few rows for {query_symbol}: {len(rows)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
        writer.writerows(rows)


def load_market_csv(path: Path) -> MarketData:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        field_map = {normalize_col(col): col for col in reader.fieldnames}
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    def col(*candidates: str, required: bool = True) -> Optional[str]:
        for candidate in candidates:
            key = normalize_col(candidate)
            if key in field_map:
                return field_map[key]
        if required:
            raise ValueError(f"Missing required column among {candidates} in {path}")
        return None

    date_col = col("date")
    open_col = col("open", required=False)
    high_col = col("high", required=False)
    low_col = col("low", required=False)
    close_col = col("adj_close", "adjusted_close", "close")
    raw_close_col = col("close", required=False)
    volume_col = col("volume", required=False)

    parsed: List[Tuple[date, float, float, float, float, float]] = []
    for row in rows:
        d = parse_date(row[date_col])
        c = parse_float(row[close_col])
        if not np.isfinite(c) or c <= 0:
            continue
        o = parse_float(row[open_col]) if open_col else c
        h = parse_float(row[high_col]) if high_col else c
        l = parse_float(row[low_col]) if low_col else c
        if not np.isfinite(o):
            o = c
        if not np.isfinite(h):
            h = c
        if not np.isfinite(l):
            l = c
        if raw_close_col and close_col != raw_close_col:
            # Keep OHLC columns coherent when adjusted close is used as close.
            adjustment_base = parse_float(row[raw_close_col])
            if np.isfinite(adjustment_base) and adjustment_base > 0:
                ratio = c / adjustment_base
                o *= ratio
                h *= ratio
                l *= ratio
        v = parse_float(row[volume_col], default=0.0) if volume_col else 0.0
        parsed.append((d, o, h, l, c, v if np.isfinite(v) else 0.0))

    parsed.sort(key=lambda x: x[0])
    dedup: Dict[date, Tuple[date, float, float, float, float, float]] = {}
    for item in parsed:
        dedup[item[0]] = item
    parsed = list(dedup.values())
    parsed.sort(key=lambda x: x[0])

    return MarketData(
        dates=[x[0] for x in parsed],
        open_=np.array([x[1] for x in parsed], dtype=float),
        high=np.array([x[2] for x in parsed], dtype=float),
        low=np.array([x[3] for x in parsed], dtype=float),
        close=np.array([x[4] for x in parsed], dtype=float),
        volume=np.array([x[5] for x in parsed], dtype=float),
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isfinite(sample).all():
            out[i] = float(np.mean(sample))
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if window <= 1 or len(values) < window:
        return out
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isfinite(sample).all():
            out[i] = float(np.std(sample, ddof=1))
    return out


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    if window <= 0 or len(values) < window:
        return out
    for i in range(window - 1, len(values)):
        sample = values[i - window + 1 : i + 1]
        if np.isfinite(sample).all():
            out[i] = float(np.max(sample))
    return out


def make_features(
    market: MarketData,
    vol_window: int,
    fast_ma: int,
    slow_ma: int,
    drawdown_window: int,
) -> FeatureData:
    close = market.close
    n = len(close)
    log_return = np.full(n, np.nan, dtype=float)
    log_return[1:] = np.log(close[1:] / close[:-1])
    return_1d = np.expm1(log_return)

    vol_ann = rolling_std(log_return, vol_window) * math.sqrt(252.0)
    ma_fast = rolling_mean(close, fast_ma)
    ma_slow = rolling_mean(close, slow_ma)
    ma_gap = np.log(ma_fast / ma_slow)
    peak = rolling_max(close, drawdown_window)
    drawdown = close / peak - 1.0

    names = ["log_return", "volatility_20d_ann", "ma_gap_20_60", "drawdown_126d"]
    values = np.column_stack([log_return, vol_ann, ma_gap, drawdown])
    return FeatureData(names, values, log_return, return_1d, vol_ann, ma_gap, drawdown)


def logsumexp(values: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.sum(stable, axis=axis, keepdims=True)
    out = max_values + np.log(summed)
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def log_gaussian_diag(x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    covars = np.maximum(covars, 1e-8)
    d = x.shape[1]
    diff = x[:, None, :] - means[None, :, :]
    term = np.sum((diff * diff) / covars[None, :, :], axis=2)
    logdet = np.sum(np.log(covars), axis=1)
    return -0.5 * (d * math.log(2.0 * math.pi) + logdet[None, :] + term)


def forward_backward(log_b: np.ndarray, pi: np.ndarray, trans: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    t_count, k_count = log_b.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_trans = np.log(np.maximum(trans, 1e-300))

    alpha = np.empty((t_count, k_count), dtype=float)
    alpha[0] = log_pi + log_b[0]
    for t in range(1, t_count):
        alpha[t] = log_b[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)

    beta = np.empty((t_count, k_count), dtype=float)
    beta[-1] = 0.0
    for t in range(t_count - 2, -1, -1):
        beta[t] = logsumexp(log_trans + log_b[t + 1][None, :] + beta[t + 1][None, :], axis=1)

    loglik = float(logsumexp(alpha[-1], axis=0))
    return alpha, beta, loglik


def initialize_params(
    x: np.ndarray,
    n_states: int,
    rng: np.random.Generator,
    randomize: bool,
) -> HMMParams:
    t_count, n_features = x.shape
    score = x[:, 0].copy()
    if x.shape[1] >= 4:
        score = x[:, 0] - 0.25 * x[:, 1] + 0.5 * x[:, 2] + 0.25 * x[:, 3]
    if randomize:
        score = score + rng.normal(0.0, 0.05, size=t_count)

    order = np.argsort(score)
    groups = np.array_split(order, n_states)
    means = np.empty((n_states, n_features), dtype=float)
    covars = np.empty((n_states, n_features), dtype=float)
    global_var = np.var(x, axis=0) + 1e-3
    for state, idx in enumerate(groups):
        if len(idx) < 2:
            means[state] = np.mean(x, axis=0) + rng.normal(0.0, 0.1, size=n_features)
            covars[state] = global_var
        else:
            means[state] = np.mean(x[idx], axis=0)
            covars[state] = np.var(x[idx], axis=0) + 1e-3

    pi = np.full(n_states, 1e-3, dtype=float)
    first_state = int(np.argmin(np.abs([np.mean(score[g]) if len(g) else 0.0 for g in groups] - score[0])))
    pi[first_state] = 1.0
    pi /= pi.sum()

    trans = np.full((n_states, n_states), 0.08 / max(n_states - 1, 1), dtype=float)
    np.fill_diagonal(trans, 0.92)
    trans /= trans.sum(axis=1, keepdims=True)

    return HMMParams(pi=pi, trans=trans, means=means, covars=covars, loglik=-np.inf)


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    n_init: int = 5,
    min_covar: float = 1e-5,
    seed: int = 42,
) -> HMMParams:
    rng = np.random.default_rng(seed)
    best: Optional[HMMParams] = None
    t_count = x.shape[0]

    for init_id in range(n_init):
        params = initialize_params(x, n_states, rng, randomize=init_id > 0)
        previous_loglik = -np.inf

        for _ in range(max_iter):
            log_b = log_gaussian_diag(x, params.means, params.covars)
            alpha, beta, loglik = forward_backward(log_b, params.pi, params.trans)
            gamma = np.exp(alpha + beta - loglik)
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            log_trans = np.log(np.maximum(params.trans, 1e-300))
            xi_sum = np.zeros_like(params.trans)
            for t in range(t_count - 1):
                log_xi = (
                    alpha[t][:, None]
                    + log_trans
                    + log_b[t + 1][None, :]
                    + beta[t + 1][None, :]
                    - loglik
                )
                xi_sum += np.exp(log_xi)

            pi = gamma[0] + 1e-8
            pi /= pi.sum()

            trans = xi_sum + 1e-8
            trans /= trans.sum(axis=1, keepdims=True)

            weights = gamma.sum(axis=0) + 1e-8
            means = (gamma.T @ x) / weights[:, None]
            covars = np.empty_like(params.covars)
            for state in range(n_states):
                diff = x - means[state]
                covars[state] = (gamma[:, state][:, None] * diff * diff).sum(axis=0) / weights[state]
            covars = np.maximum(covars, min_covar)

            params = HMMParams(pi=pi, trans=trans, means=means, covars=covars, loglik=loglik)
            if abs(loglik - previous_loglik) < tol:
                break
            previous_loglik = loglik

        if best is None or params.loglik > best.loglik:
            best = params

    assert best is not None
    return best


def viterbi(log_b: np.ndarray, pi: np.ndarray, trans: np.ndarray) -> np.ndarray:
    t_count, k_count = log_b.shape
    log_pi = np.log(np.maximum(pi, 1e-300))
    log_trans = np.log(np.maximum(trans, 1e-300))
    delta = np.empty((t_count, k_count), dtype=float)
    psi = np.empty((t_count, k_count), dtype=int)
    delta[0] = log_pi + log_b[0]
    psi[0] = 0

    for t in range(1, t_count):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = log_b[t] + np.max(scores, axis=0)

    states = np.empty(t_count, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(t_count - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def standardize_window(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=1)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def annualized_return_from_daily(mean_daily_log_return: float) -> float:
    return math.exp(mean_daily_log_return * 252.0) - 1.0


def state_statistics(
    states: np.ndarray,
    returns: np.ndarray,
    n_states: int,
) -> Dict[int, Dict[str, float]]:
    stats: Dict[int, Dict[str, float]] = {}
    finite_returns = returns[np.isfinite(returns)]
    fallback_vol = float(np.std(finite_returns, ddof=1) * math.sqrt(252.0)) if len(finite_returns) > 1 else 0.0

    for state in range(n_states):
        state_returns = returns[states == state]
        state_returns = state_returns[np.isfinite(state_returns)]
        count = int(len(state_returns))
        if count == 0:
            mean_daily = -np.inf
            ann_return = -np.inf
            ann_vol = fallback_vol
            sharpe = -np.inf
        else:
            mean_daily = float(np.mean(state_returns))
            ann_return = annualized_return_from_daily(mean_daily)
            ann_vol = float(np.std(state_returns, ddof=1) * math.sqrt(252.0)) if count > 1 else fallback_vol
            sharpe = ann_return / ann_vol if ann_vol > 1e-12 else np.sign(ann_return) * np.inf
        stats[state] = {
            "count": count,
            "mean_daily_return": mean_daily,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
        }
    return stats


def map_states_to_labels(stats: Dict[int, Dict[str, float]]) -> Dict[int, str]:
    states = list(stats.keys())

    def rank_key(state: int) -> Tuple[float, float]:
        s = stats[state]
        return (float(s["sharpe"]), float(s["ann_return"]))

    bull_state = max(states, key=rank_key)
    bear_state = min(states, key=rank_key)
    mapping: Dict[int, str] = {}
    for state in states:
        if state == bull_state:
            mapping[state] = "Bull"
        elif state == bear_state:
            mapping[state] = "Bear"
        else:
            mapping[state] = "Neutral"
    return mapping


def run_rolling_hmm(
    features: FeatureData,
    train_window: int,
    fit_step: int,
    n_states: int,
    max_iter: int,
    n_init: int,
    seed: int,
) -> List[RegimePoint]:
    feature_values = features.values
    valid_mask = np.isfinite(feature_values).all(axis=1) & np.isfinite(features.log_return)
    valid_indices = np.flatnonzero(valid_mask)
    if len(valid_indices) < train_window:
        raise ValueError(
            f"Not enough valid observations for train_window={train_window}. "
            f"Valid observations: {len(valid_indices)}"
        )

    points: List[RegimePoint] = []
    scheduled_positions = list(range(train_window - 1, len(valid_indices), fit_step))
    if scheduled_positions[-1] != len(valid_indices) - 1:
        scheduled_positions.append(len(valid_indices) - 1)

    for job_id, pos in enumerate(scheduled_positions):
        window_valid_indices = valid_indices[pos - train_window + 1 : pos + 1]
        original_idx = int(window_valid_indices[-1])
        x_raw = feature_values[window_valid_indices]
        x, _, _ = standardize_window(x_raw)

        params = fit_gaussian_hmm(
            x,
            n_states=n_states,
            max_iter=max_iter,
            n_init=n_init,
            seed=seed + job_id,
        )
        log_b = log_gaussian_diag(x, params.means, params.covars)
        alpha, beta, loglik = forward_backward(log_b, params.pi, params.trans)
        gamma = np.exp(alpha + beta - loglik)
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        state_seq = viterbi(log_b, params.pi, params.trans)

        returns_window = features.log_return[window_valid_indices]
        stats = state_statistics(state_seq, returns_window, n_states)
        label_mapping = map_states_to_labels(stats)

        current_state = int(state_seq[-1])
        current_label = label_mapping[current_state]
        mapped_probs = {"Bear": 0.0, "Neutral": 0.0, "Bull": 0.0}
        for state in range(n_states):
            mapped_probs[label_mapping[state]] += float(gamma[-1, state])

        current_stats = stats[current_state]
        points.append(
            RegimePoint(
                idx=original_idx,
                raw_state=current_state,
                label=current_label,
                prob_bear=mapped_probs["Bear"],
                prob_neutral=mapped_probs["Neutral"],
                prob_bull=mapped_probs["Bull"],
                state_mean_daily_return=float(current_stats["mean_daily_return"]),
                state_ann_return=float(current_stats["ann_return"]),
                state_ann_vol=float(current_stats["ann_vol"]),
                state_sharpe=float(current_stats["sharpe"]),
                state_count=int(current_stats["count"]),
                model_loglik=float(loglik),
            )
        )

        if (job_id + 1) % 10 == 0 or job_id == len(scheduled_positions) - 1:
            print(
                f"[{job_id + 1}/{len(scheduled_positions)}] "
                f"labeled index={original_idx} label={current_label}",
                file=sys.stderr,
            )

    return points


def smooth_probabilities(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    out = np.full(values.shape, np.nan, dtype=float)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        sample = values[start : i + 1]
        sample = sample[np.isfinite(sample)]
        if len(sample):
            out[i] = float(np.mean(sample))
    return out


def write_labeled_csv(
    out_path: Path,
    market: MarketData,
    features: FeatureData,
    points: Sequence[RegimePoint],
    target_horizon: int,
    smoothing_window: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    point_by_idx = {p.idx: p for p in points}
    point_indices = [p.idx for p in points]

    labels = [point_by_idx[idx].label for idx in point_indices]
    future_label_by_idx: Dict[int, str] = {}
    if target_horizon > 0:
        for i, idx in enumerate(point_indices):
            future_pos = i + target_horizon
            if future_pos < len(point_indices):
                future_label_by_idx[idx] = labels[future_pos]

    prob_bear = np.array([point_by_idx[idx].prob_bear for idx in point_indices], dtype=float)
    prob_neutral = np.array([point_by_idx[idx].prob_neutral for idx in point_indices], dtype=float)
    prob_bull = np.array([point_by_idx[idx].prob_bull for idx in point_indices], dtype=float)
    smooth_bear = smooth_probabilities(prob_bear, smoothing_window)
    smooth_neutral = smooth_probabilities(prob_neutral, smoothing_window)
    smooth_bull = smooth_probabilities(prob_bull, smoothing_window)

    header = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "log_return",
        "volatility_20d_ann",
        "ma_gap_20_60",
        "drawdown_126d",
        "hmm_state",
        "hmm_label",
        "hmm_label_code",
        "prob_bear",
        "prob_neutral",
        "prob_bull",
        f"smooth_prob_bear_{smoothing_window}",
        f"smooth_prob_neutral_{smoothing_window}",
        f"smooth_prob_bull_{smoothing_window}",
        "state_mean_daily_return",
        "state_ann_return",
        "state_ann_vol",
        "state_sharpe",
        "state_count",
        "model_loglik",
    ]
    if target_horizon > 0:
        header.extend([f"target_label_plus_{target_horizon}_steps", f"target_code_plus_{target_horizon}_steps"])

    def fmt(value: float) -> str:
        if not np.isfinite(value):
            return ""
        return f"{value:.10g}"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_id, idx in enumerate(point_indices):
            point = point_by_idx[idx]
            row = [
                market.dates[idx].isoformat(),
                fmt(market.open_[idx]),
                fmt(market.high[idx]),
                fmt(market.low[idx]),
                fmt(market.close[idx]),
                fmt(market.volume[idx]),
                fmt(features.return_1d[idx]),
                fmt(features.log_return[idx]),
                fmt(features.vol_ann[idx]),
                fmt(features.ma_gap[idx]),
                fmt(features.drawdown[idx]),
                point.raw_state,
                point.label,
                LABEL_TO_CODE[point.label],
                fmt(point.prob_bear),
                fmt(point.prob_neutral),
                fmt(point.prob_bull),
                fmt(smooth_bear[row_id]),
                fmt(smooth_neutral[row_id]),
                fmt(smooth_bull[row_id]),
                fmt(point.state_mean_daily_return),
                fmt(point.state_ann_return),
                fmt(point.state_ann_vol),
                fmt(point.state_sharpe),
                point.state_count,
                fmt(point.model_loglik),
            ]
            if target_horizon > 0:
                target_label = future_label_by_idx.get(idx, "")
                row.extend([target_label, LABEL_TO_CODE[target_label] if target_label else ""])
            writer.writerow(row)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rolling Gaussian HMM market regime labeling")
    parser.add_argument("--input", type=Path, default=None, help="Input OHLCV CSV. If omitted, Stooq download is used.")
    parser.add_argument("--symbol", default="SPY.US", help="Stooq symbol used when --input is omitted. Example: SPY.US")
    parser.add_argument("--start", default="2010-01-01", help="Download start date, YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="Download end date, YYYY-MM-DD")
    parser.add_argument("--raw-out", type=Path, default=Path("data/raw/spy_stooq_daily.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/spy_hmm_regime_labels.csv"))
    parser.add_argument("--train-window", type=int, default=504, help="Rolling HMM fitting window in valid trading days")
    parser.add_argument("--fit-step", type=int, default=20, help="Fit and label every N valid trading days")
    parser.add_argument("--states", type=int, default=3, help="Number of HMM hidden states")
    parser.add_argument("--max-iter", type=int, default=80, help="EM iterations per HMM fit")
    parser.add_argument("--n-init", type=int, default=4, help="Random restarts per HMM fit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--fast-ma", type=int, default=20)
    parser.add_argument("--slow-ma", type=int, default=60)
    parser.add_argument("--drawdown-window", type=int, default=126)
    parser.add_argument("--smoothing-window", type=int, default=3, help="Smoothing over labeled rows, not calendar days")
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=1,
        help="Future labeled-row horizon for deep-learning target columns. 1 means next rebalance label.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.input is None:
        print(f"Downloading {args.symbol} to {args.raw_out}", file=sys.stderr)
        try:
            download_yahoo_chart(args.symbol, args.start, args.end, args.raw_out)
        except Exception as yahoo_error:
            print(f"Yahoo download failed: {yahoo_error}", file=sys.stderr)
            print("Trying Stooq fallback", file=sys.stderr)
            download_stooq(args.symbol, args.start, args.end, args.raw_out)
        input_path = args.raw_out
    else:
        input_path = args.input

    market = load_market_csv(input_path)
    if len(market.dates) < args.train_window + args.drawdown_window:
        raise ValueError(
            f"Not enough rows: {len(market.dates)}. "
            f"Need comfortably more than train_window + warmup windows."
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
        args.output,
        market,
        features,
        points,
        target_horizon=args.target_horizon,
        smoothing_window=args.smoothing_window,
    )

    counts: Dict[str, int] = {"Bear": 0, "Neutral": 0, "Bull": 0}
    for point in points:
        counts[point.label] += 1
    print(f"Wrote {len(points)} labeled rows to {args.output}", file=sys.stderr)
    print(f"Label counts: {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
