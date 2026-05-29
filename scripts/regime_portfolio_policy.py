"""Regime-aware portfolio policy for the final momentum-tilt backtest.

The classifier is unchanged. This module only translates model probabilities
[p_bear, p_neutral, p_bull] into portfolio weights over:

    [SPY, QQQ, GLD, TLT, CASH]

The final policy is intentionally return-seeking: it uses SPY/QQQ as risk-on
assets and moves part of the portfolio to GLD/TLT/CASH only when the model or
trend/drawdown guards call for defense.
"""

from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np


ASSETS = ("SPY", "QQQ", "GLD", "TLT")
ASSET_COLS = ASSETS + ("CASH",)


def _safe_normalize(w: Sequence[float], fallback: Sequence[float] | None = None) -> np.ndarray:
    arr = np.asarray(w, dtype=float).copy()
    arr[~np.isfinite(arr)] = 0.0
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0.0:
        if fallback is None:
            return np.full_like(arr, 1.0 / len(arr), dtype=float)
        return _safe_normalize(fallback)
    return arr / total


def project_long_only_with_caps(w: Sequence[float], caps: Sequence[float]) -> np.ndarray:
    """Project long-only weights to sum=1 while respecting simple max caps."""
    weights = _safe_normalize(w)
    caps_arr = np.asarray(caps, dtype=float)
    if caps_arr.shape != weights.shape:
        raise ValueError("caps must have the same shape as weights")
    if caps_arr.sum() < 1.0 - 1e-12:
        raise ValueError("sum(caps) must be at least 1")

    fixed = np.zeros(len(weights), dtype=bool)
    for _ in range(20):
        over = (weights > caps_arr + 1e-12) & (~fixed)
        if not np.any(over):
            break

        excess = float(np.sum(weights[over] - caps_arr[over]))
        weights[over] = caps_arr[over]
        fixed |= over

        free = ~fixed
        if not np.any(free):
            break

        base = weights[free].copy()
        base = base / base.sum() if base.sum() > 0.0 else np.full_like(base, 1.0 / len(base))
        weights[free] += excess * base

    return _safe_normalize(np.minimum(weights, caps_arr))


def get_period_return(prices: Mapping[str, float], start_date: str, end_date: str) -> float:
    if start_date not in prices or end_date not in prices:
        return 0.0
    p0 = float(prices[start_date])
    p1 = float(prices[end_date])
    if p0 <= 0.0 or p1 <= 0.0:
        return 0.0
    return p1 / p0 - 1.0


def _sma(prices: Mapping[str, float], dates: Sequence[str], end_date: str, window: int) -> float | None:
    if end_date not in prices:
        return None
    try:
        idx = dates.index(end_date)
    except ValueError:
        return None
    if idx + 1 < window:
        return None

    vals = np.array([prices[d] for d in dates[idx - window + 1 : idx + 1]], dtype=float)
    return float(np.mean(vals))


def ma_gap(prices: Mapping[str, float], dates: Sequence[str], end_date: str, fast: int, slow: int) -> float:
    fast_sma = _sma(prices, dates, end_date, fast)
    slow_sma = _sma(prices, dates, end_date, slow)
    if fast_sma is None or slow_sma is None or slow_sma <= 0.0:
        return 0.0
    return fast_sma / slow_sma - 1.0


def rolling_drawdown(prices: Mapping[str, float], dates: Sequence[str], end_date: str, window: int) -> float:
    if end_date not in prices:
        return 0.0
    try:
        idx = dates.index(end_date)
    except ValueError:
        return 0.0

    start = max(0, idx - window + 1)
    vals = np.array([prices[d] for d in dates[start : idx + 1]], dtype=float)
    peak = float(np.max(vals)) if len(vals) else 0.0
    if peak <= 0.0:
        return 0.0
    return float(vals[-1] / peak - 1.0)


def sorted_common_dates(prices_by_asset: Mapping[str, Mapping[str, float]]) -> List[str]:
    common = None
    for prices in prices_by_asset.values():
        keys = set(prices.keys())
        common = keys if common is None else common & keys
    return sorted(common or [])


def compute_return_seeking_weights(
    probs: Sequence[float],
    prices_by_asset: Mapping[str, Mapping[str, float]],
    asof_date: str,
    prev_w: np.ndarray | None = None,
) -> np.ndarray:
    """Final no-lead regime momentum policy.

    Bull/neutral regimes push the portfolio toward SPY/QQQ. Bear probability,
    weak trend, and large SPY drawdowns cap equity exposure and add GLD/TLT/CASH.
    """
    p_bear, p_neutral, p_bull = _safe_normalize(probs, fallback=np.ones(3) / 3)

    spy_prices = prices_by_asset["SPY"]
    qqq_prices = prices_by_asset["QQQ"]
    spy_dates = sorted(spy_prices.keys())
    qqq_dates = sorted(qqq_prices.keys())

    spy_trend = ma_gap(spy_prices, spy_dates, asof_date, fast=20, slow=60)
    qqq_trend = ma_gap(qqq_prices, qqq_dates, asof_date, fast=20, slow=60)
    spy_dd = rolling_drawdown(spy_prices, spy_dates, asof_date, window=126)

    equity = 0.25 + 1.15 * p_bull + 0.55 * p_neutral - 0.35 * p_bear
    if spy_trend > 0.0:
        equity += 0.12
    if qqq_trend > 0.0:
        equity += 0.06
    if spy_trend < 0.0:
        equity -= 0.20

    if spy_dd < -0.10:
        equity = min(equity, 0.55)
    if spy_dd < -0.15:
        equity = min(equity, 0.35)
    if p_bear > max(p_neutral, p_bull):
        equity = min(equity, 0.60)

    equity = float(np.clip(equity * 1.20, 0.20, 0.98))

    qqq_share = 0.55 + 0.25 * (p_bull - p_bear)
    if qqq_trend > spy_trend:
        qqq_share += 0.10
    qqq_share = float(np.clip(qqq_share, 0.35, 0.80))

    target = np.zeros(len(ASSET_COLS), dtype=float)
    target[0] = equity * (1.0 - qqq_share)
    target[1] = equity * qqq_share

    if spy_trend < 0.0 or p_bear > p_bull:
        defense = min(0.18, max(0.0, 1.0 - equity - 0.02))
        target[2] = defense * 0.35
        target[3] = defense * 0.65

    target[4] = max(0.0, 1.0 - float(target[:4].sum()))
    target = project_long_only_with_caps(target, [0.65, 0.80, 0.25, 0.40, 0.80])

    if prev_w is not None:
        target = 0.75 * target + 0.25 * _safe_normalize(prev_w)
        target = project_long_only_with_caps(target, [0.65, 0.80, 0.25, 0.40, 0.80])

    return target


__all__ = [
    "ASSETS",
    "ASSET_COLS",
    "compute_return_seeking_weights",
    "get_period_return",
]
