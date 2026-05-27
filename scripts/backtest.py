"""
백테스트: 모델 포트폴리오 vs Baseline 전략 비교
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import RegimeClassifier

device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)


# ── 1. 데이터 로드 ───────────────────────────────────────────────
data   = np.load("data/processed/cross_asset_supervised_30d_5d.npz", allow_pickle=True)
X_test = torch.tensor(data["X_test"].astype(np.float32)).to(device)

# 날짜 인덱스
index_rows = list(csv.DictReader(open("data/processed/cross_asset_supervised_30d_5d_index.csv")))
test_rows  = [r for r in index_rows if r["split"] == "test"]   # 105개

# SPY 일별 가격 (Adj Close)
spy_prices = {}   # {date_str: adj_close}
with open("data/raw/spy_daily.csv") as f:
    for row in csv.DictReader(f):
        spy_prices[row["Date"]] = float(row["Adj Close"])

spy_dates  = sorted(spy_prices.keys())


# ── 2. 모델 예측 ─────────────────────────────────────────────────
model = RegimeClassifier(input_size=40, conv_channels=16, lstm_hidden=32).to(device)
model.load_state_dict(torch.load("outputs/models/best_model.pt", map_location=device))
model.eval()

with torch.no_grad():
    probs = model.predict_proba(X_test).cpu().numpy()   # (105, 3): [p_bear, p_neutral, p_bull]

# 모델 포트폴리오 비중: Bull 확률 + Neutral의 절반
w_model = probs[:, 2] + 0.5 * probs[:, 1]   # (105,)


# ── 3. 각 holding period의 SPY 수익률 계산 ──────────────────────
def get_period_return(start_date: str, end_date: str) -> float:
    """start_date 종가 → end_date 종가 단순 수익률"""
    if start_date not in spy_prices or end_date not in spy_prices:
        return 0.0
    return spy_prices[end_date] / spy_prices[start_date] - 1.0

# holding period: input_end_date → target_date
holding_returns = []
for r in test_rows:
    ret = get_period_return(r["input_end_date"], r["target_date"])
    holding_returns.append(ret)
holding_returns = np.array(holding_returns)   # (105,)

# 해당 기간 SPY 변동성 (20일 rolling → 연율화, baseline용)
def get_rolling_vol(end_date: str, window: int = 20) -> float:
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window:
        return 0.15   # fallback
    recent = spy_dates[idx - window: idx + 1]
    closes = [spy_prices[d] for d in recent]
    log_rets = [np.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    return np.std(log_rets) * np.sqrt(252)

# SMA 계산 (MA Crossover baseline용)
def get_sma(end_date: str, window: int) -> float:
    idx = spy_dates.index(end_date) if end_date in spy_dates else -1
    if idx < window:
        return spy_prices[end_date]
    recent = spy_dates[idx - window + 1: idx + 1]
    return np.mean([spy_prices[d] for d in recent])


# ── 4. Baseline 전략 비중 계산 ───────────────────────────────────
TARGET_VOL = 0.15   # 변동성 타게팅 목표 (연율화 15%)

w_strategies = {}
w_strategies["Buy & Hold"]          = np.ones(105)
w_strategies["60/40"]               = np.full(105, 0.60)
w_strategies["80/20"]               = np.full(105, 0.80)
w_strategies["40/60"]               = np.full(105, 0.40)
w_strategies["Conv1D+LSTM (ours)"]  = w_model

# MA Crossover: SMA20 > SMA60 → 전액 주식, 아니면 전액 현금
ma_weights = []
for r in test_rows:
    sma20 = get_sma(r["input_end_date"], 20)
    sma60 = get_sma(r["input_end_date"], 60)
    ma_weights.append(1.0 if sma20 > sma60 else 0.0)
w_strategies["MA Crossover"] = np.array(ma_weights)

# Volatility Targeting: 목표변동성 / 실현변동성 (최대 1.0)
vol_weights = []
for r in test_rows:
    rv = get_rolling_vol(r["input_end_date"], 20)
    vol_weights.append(min(TARGET_VOL / rv, 1.0) if rv > 0 else 1.0)
w_strategies["Vol Targeting"] = np.array(vol_weights)


# ── 5. 성과 계산 함수 ────────────────────────────────────────────
TRANSACTION_COST = 0.001   # 편도 0.1%
RISK_FREE_RATE   = 0.05    # 연 5% (최근 미국 금리 기준)
PERIODS_PER_YEAR = 252 / 5 # 5거래일 단위 → 연간 50.4 periods

def compute_metrics(weights: np.ndarray, holding_rets: np.ndarray, name: str) -> dict:
    # 거래비용: 비중 변화량 × 0.1%
    w_prev = np.concatenate([[0.0], weights[:-1]])
    turnover    = np.abs(weights - w_prev)
    cost        = turnover * TRANSACTION_COST

    # 기간별 포트폴리오 수익률
    port_rets = weights * holding_rets - cost

    # 누적 수익률
    cum_ret = np.prod(1 + port_rets) - 1

    # 연율화 수익률 / 변동성
    ann_ret = (1 + cum_ret) ** (PERIODS_PER_YEAR / len(port_rets)) - 1
    ann_vol = port_rets.std() * np.sqrt(PERIODS_PER_YEAR)

    # Sharpe Ratio
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0.0

    # Max Drawdown
    cum_curve = np.cumprod(1 + port_rets)
    running_max = np.maximum.accumulate(cum_curve)
    drawdowns = cum_curve / running_max - 1
    mdd = drawdowns.min()

    # Calmar Ratio
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0

    return {
        "name":      name,
        "cum_ret":   cum_ret,
        "ann_ret":   ann_ret,
        "ann_vol":   ann_vol,
        "sharpe":    sharpe,
        "mdd":       mdd,
        "calmar":    calmar,
        "port_rets": port_rets,
        "weights":   weights,
    }


# ── 6. 결과 출력 ─────────────────────────────────────────────────
results = {}
for name, weights in w_strategies.items():
    results[name] = compute_metrics(weights, holding_returns, name)

# 테이블 출력
header = f"{'전략':<22} {'누적수익':>8} {'연수익':>7} {'변동성':>7} {'Sharpe':>7} {'MDD':>8} {'Calmar':>7}"
print(f"\n{'='*74}")
print(f"백테스트 기간: {test_rows[0]['input_end_date']} ~ {test_rows[-1]['target_date']}")
print(f"{'='*74}")
print(header)
print("─" * 74)

order = ["Buy & Hold", "60/40", "80/20", "40/60",
         "MA Crossover", "Vol Targeting", "Conv1D+LSTM (ours)"]
for name in order:
    m = results[name]
    marker = " ◀" if name == "Conv1D+LSTM (ours)" else ""
    print(f"{name:<22} {m['cum_ret']:>7.1%}  {m['ann_ret']:>6.1%}  "
          f"{m['ann_vol']:>6.1%}  {m['sharpe']:>6.2f}  {m['mdd']:>7.1%}  "
          f"{m['calmar']:>6.2f}{marker}")

print("─" * 74)
print(f"  * 거래비용 편도 {TRANSACTION_COST*100:.1f}% 반영, 무위험수익률 {RISK_FREE_RATE*100:.0f}%")

# 결과 저장
save = {k: {kk: float(vv) for kk, vv in v.items()
            if kk not in ("port_rets", "weights", "name")}
        for k, v in results.items()}
with open("outputs/results/backtest_results.json", "w") as f:
    json.dump(save, f, indent=2)
print("\n결과 저장: outputs/results/backtest_results.json")


if __name__ == "__main__":
    pass
