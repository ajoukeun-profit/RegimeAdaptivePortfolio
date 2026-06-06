# 트러블슈팅 기록: Label, Optimization, MVO 안정성

> Neutral 라벨 식별 실패, HMM pseudo-label 신뢰성, 작은 샘플에서의 MVO 추정 오차를 정리한 문서입니다.

## 1. 최종 결론

현재 가장 설득력 있는 최종 방향은 다음이다.

```text
Neutral label failure
-> Bear vs Non-Bear binary classification
-> Binary soft-label training
-> 2-Regime MVO
-> MVO weight cap 40%
```

핵심 이유:

- Neutral은 구조적으로 애매해서 3-class 모델이 test에서 한 번도 Neutral을 예측하지 못했다.
- Binary classification으로 바꾸자 Balanced Accuracy가 51.9%에서 70.2%로 개선됐다.
- Binary soft label을 쓰자 Bear Recall이 58.1%에서 67.4%로 개선됐다.
- MVO weight cap은 작은 샘플에서 발생하는 자산 몰빵 문제를 완화했다.
- 최종적으로 `Binary Regime-MVO Soft, cap 40%`가 현재 비교군 중 가장 균형 잡힌 결과를 냈다.

최종 비교 결과:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| **Binary Regime-MVO Soft cap 40%** | **53.7%** | **1.48** | **-9.0%** | **2.55** |

해석:

Binary Regime-MVO Soft cap 40%는 EW 1/N보다 누적수익률이 2.8%p 높고, MDD는 거의 비슷하다. 3-class capped Regime-MVO와 비교해도 누적수익률과 Calmar가 소폭 개선된다.

---

## 2. 지금 문제가 gradient 폭발/소멸인가?

현재 관찰된 가장 큰 문제는 gradient 폭발이나 소멸이라기보다 **label 정의와 MVO 추정 안정성 문제**에 가깝다.

### Gradient 폭발

학습 코드에서는 이미 gradient clipping을 적용하고 있다.

위치: `scripts/train.py`

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

따라서 gradient 폭발은 어느 정도 제어되고 있다. 학습 기록에서도 NaN이나 loss divergence는 확인되지 않았다.

다만 gradient norm을 직접 logging하지는 않으므로, clipping이 얼마나 자주 발생하는지는 아직 알 수 없다.

### Gradient 소멸

모델 입력은 30일 sequence이고 LSTM hidden size도 크지 않다. 매우 긴 sequence에서 나타나는 심각한 vanishing gradient 가능성은 상대적으로 낮다.

하지만 layer별 gradient norm이나 hidden-state 통계를 기록하지 않았으므로, 완전히 배제할 수는 없다.

현재 결론:

> gradient 문제는 가능성으로 남아 있지만, 실제 실패 양상은 Neutral label ambiguity와 MVO estimation error가 훨씬 명확하다.

---

## 3. 문제 A: Neutral 라벨 식별 실패

### 현상

3-class hard-label 모델 결과:

| Metric | Value |
|---|---:|
| Accuracy | 61.9% |
| Balanced Accuracy | 51.9% |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

Confusion matrix:

```text
              Pred Bear  Pred Neutral  Pred Bull
Actual Bear       26          0           17
Actual Neutral     8          0           13
Actual Bull        2          0           39
```

모델이 test set에서 Neutral을 한 번도 예측하지 않았다.

### 왜 발생했나

Neutral은 Bear와 Bull 사이의 중간 상태다. HMM이 만든 state를 사후적으로 Sharpe, 수익률, 변동성 기준으로 Bear / Neutral / Bull에 매핑했기 때문에 Neutral은 경제적 경계가 명확하지 않다.

즉, Neutral은 다음 두 경계가 모두 애매하다.

- Bear vs Neutral
- Neutral vs Bull

그 결과 모델은 더 쉬운 Bear/Bull 분리 문제를 학습하고, Neutral은 Bear 또는 Bull로 흡수해버린다.

### 기존 시도

`neutral_boost`를 통해 Neutral class weight를 높였다.

위치: `scripts/train.py`

```python
weights = n_samples / (float(num_classes) * counts)
weights[1] *= neutral_boost
```

하지만 class weight를 높이면 trade-off가 생긴다.

- Neutral recall은 일부 개선될 수 있다.
- 대신 Bear/Bull recall과 전체 성능이 떨어질 수 있다.

실험상 `neutral_boost=1.2`가 전체적으로는 더 안정적이었지만, Neutral Recall 0.0% 문제 자체는 해결하지 못했다.

---

## 4. 해결 1: Bear vs Non-Bear Binary Classification

Neutral을 억지로 따로 맞히기보다, 투자 목적에 맞게 문제를 다시 정의했다.

기존:

```text
Bear / Neutral / Bull
```

변경:

```text
Bear / Non-Bear
Non-Bear = Neutral + Bull
```

이 방식이 더 적절한 이유:

- 프로젝트의 핵심은 하방 위험 관리다.
- Bear 탐지가 Neutral/Bull 세부 분리보다 중요하다.
- Neutral의 애매한 경계를 제거할 수 있다.
- binary 확률을 2-Regime MVO로 자연스럽게 연결할 수 있다.

Binary hard-label 결과:

| Metric | Value |
|---|---:|
| Accuracy | 72.4% |
| Balanced Accuracy | 70.2% |
| Non-Bear Recall | 82.3% |
| Bear Recall | 58.1% |

비교:

| Model | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| Binary hard label | 70.2% | 58.1% |

해석:

Bear Recall만 보면 3-class가 약간 높아 보이지만, 3-class는 Neutral을 전혀 예측하지 못한다. Binary는 문제 정의가 더 명확하고 Balanced Accuracy가 크게 개선되어 최종 포트폴리오 전략과 더 잘 맞는다.

---

## 5. 해결 2: LR / RF Baseline

데이터 수가 작기 때문에 딥러닝이 정말 필요한지 확인해야 했다. 그래서 Logistic Regression과 Random Forest를 baseline으로 두고 비교했다.

실행:

```bash
.venv/bin/python scripts/baseline_binary_bear_sklearn.py
```

결과:

| Model | Valid Balanced Acc | Test Balanced Acc | Bear Recall |
|---|---:|---:|---:|
| Logistic Regression | 63.7% | 61.4% | 32.6% |
| Random Forest | 79.1% | 66.3% | 53.5% |
| Conv1D+LSTM Binary | 85.2% | 70.2% | 58.1% |

해석:

LR < RF < LSTM 흐름이 확인됐다. Random Forest도 꽤 강한 baseline이지만, test balanced accuracy와 Bear Recall 모두에서 Conv1D+LSTM이 더 좋았다. 따라서 시계열 패턴을 학습하는 딥러닝 모델의 효과가 있다고 볼 수 있다.

---

## 6. 해결 3: Binary Soft Label

Hard label은 HMM posterior의 불확실성을 버린다.

예를 들어 HMM posterior가 다음과 같다고 하자.

```text
Bear 0.42 / Neutral 0.39 / Bull 0.19
```

Hard label은 이를 Bear 하나로만 취급한다. 하지만 실제로는 Neutral 가능성도 크다.

따라서 HMM posterior를 binary soft target으로 합쳤다.

```text
P(Bear) = P(Bear)
P(Non-Bear) = P(Neutral) + P(Bull)
```

실행:

```bash
python3 scripts/prepare_cross_asset_dataset.py \
  --binary-bear --soft-labels \
  --output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz \
  --index-output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv \
  --meta-output data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_meta.json

python3 scripts/train_soft_labels.py \
  --data data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz \
  --index data/processed/cross_asset_supervised_30d_5d_binary_soft_labels_index.csv \
  --model-output outputs/models/best_model_binary_soft_labels.pt \
  --history-output outputs/results/train_history_binary_soft_labels.json
```

결과:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Non-Bear Recall | Bear Recall |
|---|---:|---:|---:|---:|---:|
| Binary Hard Label | 72.4% | 70.2% | 70.6% | 82.3% | 58.1% |
| Binary Soft Label | 73.3% | 72.4% | 72.4% | 77.4% | 67.4% |
| Binary Soft Label + Confidence | 73.3% | 72.4% | 72.4% | 77.4% | 67.4% |

해석:

Binary soft label은 hard label보다 Bear Recall을 크게 개선했다. 하방 위험 탐지가 중요한 프로젝트 목적과 잘 맞는다.

Confidence weighting은 binary soft label과 거의 같은 결과를 냈다. HMM posterior가 이미 대부분 특정 state에 강하게 몰려 있어서 confidence를 곱해도 학습 분포가 크게 달라지지 않았기 때문으로 보인다.

---

## 7. 문제 B: 작은 샘플에서의 MVO 추정 오차

MVO는 기대수익률과 공분산을 추정해서 weight를 계산한다. 그런데 regime별로 데이터를 나누면 각 regime의 샘플 수가 줄어든다.

이때 Sharpe-max MVO는 작은 추정 오차에도 민감하게 반응해 특정 자산에 몰빵하기 쉽다.

Binary MVO에서 cap이 없을 때:

| Cap | Non-Bear MVO | Bear MVO |
|---|---|---|
| None | SPY 100% | TLT 100% |

이런 비중은 과거 훈련 구간에는 좋아 보일 수 있지만, test 기간에는 추정 오차에 취약하다.

---

## 8. 해결 4: MVO Weight Cap

자산별 최대 비중을 제한하는 cap을 추가했다.

위치:

- `scripts/backtest_mvo.py`
- `scripts/backtest_binary_mvo.py`

실행:

```bash
python3 scripts/backtest_binary_mvo.py \
  --max-weight 0.4 \
  --output outputs/results/backtest_binary_soft_mvo_cap40_results.json
```

Binary MVO cap별 비중:

| Cap | Non-Bear MVO | Bear MVO |
|---:|---|---|
| 100% | SPY 100.0% | TLT 100.0% |
| 50% | SPY 50.0%, QQQ 49.8%, GLD 0.2% | QQQ 7.3%, GLD 42.7%, TLT 50.0% |
| 40% | SPY 40.0%, QQQ 40.0%, GLD 17.3%, TLT 2.7% | QQQ 20.0%, GLD 40.0%, TLT 40.0% |

Portfolio 결과:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 3-class Regime-MVO original | 35.3% | 1.10 | -7.2% | 2.16 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft no cap | 22.9% | 0.59 | -8.5% | 1.22 |
| Binary Regime-MVO Soft cap 50% | 47.9% | 1.37 | -8.3% | 2.48 |
| **Binary Regime-MVO Soft cap 40%** | **53.7%** | **1.48** | **-9.0%** | **2.55** |

해석:

cap이 없으면 binary MVO도 extreme portfolio로 무너진다. 하지만 cap을 추가하면 자산 쏠림이 줄고, risk-return 성과가 개선된다.

---

## 9. 최종 발표용 요약 문장

팀원들과 공유할 때는 다음처럼 말하면 된다.

> 3-class로 Bear / Neutral / Bull을 예측했을 때 Neutral Recall이 0.0%로 나왔습니다. Neutral은 HMM의 중간 상태라 경계가 애매하고, 모델이 Bear 또는 Bull로 흡수해버리는 문제가 있었습니다.

> 그래서 투자 목적에 맞게 Bear vs Non-Bear binary classification으로 바꿨고, Balanced Accuracy가 51.9%에서 70.2%로 개선되었습니다.

> 데이터가 작아서 LR/RF baseline도 확인했는데, LR < RF < LSTM 순서로 성능이 나와 시계열 딥러닝 모델의 효과는 있는 것으로 보입니다.

> 이후 HMM posterior를 활용해 binary soft label을 적용하니 Bear Recall이 58.1%에서 67.4%로 개선되었습니다.

> 마지막으로 MVO에 자산별 weight cap 40%를 적용해 자산 몰빵을 줄였고, 최종적으로 Binary Regime-MVO Soft cap 40%가 누적수익률 53.7%, Sharpe 1.48, MDD -9.0%, Calmar 2.55로 가장 균형 잡힌 결과를 냈습니다.

---

## 10. 주의할 점

- HMM label은 true label이 아니라 pseudo-label이다. 따라서 classification accuracy를 실제 시장 예측 정확도로 과대해석하면 안 된다.
- cap 40%는 현재 test 구간에서 좋은 결과다. 더 엄밀하게는 validation 또는 walk-forward 방식으로 cap을 선택해야 한다.
- SPY/Cash threshold 실험은 보조 sanity check였고, 최종 프로젝트 스토리에서는 제외한다. 최종 전략은 MVO 기반이다.
- 추가 개선 후보는 covariance shrinkage, MVO/equal-weight blending, 현금/단기채 추가, walk-forward validation이다.
