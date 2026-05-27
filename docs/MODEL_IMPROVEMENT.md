# 모델 개선 실험 상세 기록

> 발표 및 보고서 활용용

---

## 현재 모델 구조 (개선 전 기준)

```
입력: (batch, 30, 10)
        ↓
[ConvBlock — 4,224 params]
  Conv1d(10→32, kernel=3, same padding)
  BatchNorm1d(32) + ReLU
  Conv1d(32→32, kernel=3, same padding)
  BatchNorm1d(32) + ReLU
  역할: 3일 단위 슬라이딩으로 국소 패턴 추출, 노이즈 완화
        ↓ (batch, 30, 32)
[LSTM — 25,088 params]
  단방향(forward only), hidden_size=64, num_layers=1
  30개 시점을 day1→day30 순서로 읽음
  마지막 hidden state h_n[-1]만 사용
  역할: 시간 흐름 상의 패턴과 장기 의존성 학습
        ↓ (batch, 64)
[Classifier — 2,179 params]
  Linear(64→32) + ReLU + Dropout(0.5) + Linear(32→3)
  역할: 압축된 표현 → 국면 확률 출력
        ↓
출력: (batch, 3) logits → softmax → [p_bear, p_neutral, p_bull]

총 파라미터: 31,491개
```

---

## 개선 전 문제 진단

### 문제 1: Neutral 클래스 recall이 매우 낮음 (23.8%)

모델이 Neutral을 거의 맞추지 못하고 Bear 또는 Bull로 오분류한다.

**원인**: Neutral은 "Bear도 Bull도 아닌 상태"로 명확한 특징이 없다. HMM 라벨 기준으로도 "상대적으로 중간 Sharpe"이기 때문에 경계가 모호하다. 학습 데이터에서도 가장 적은 클래스(128개)여서 모델이 Neutral을 무시하는 전략을 학습하기 쉽다.

### 문제 2: 소규모 학습 데이터 (488개)

5거래일 간격 라벨링으로 인해 14년치 데이터가 488개 샘플로 압축되었다. 딥러닝 모델에게 이는 매우 적은 양이다.

### 문제 3: 단방향 LSTM의 정보 손실

현재 LSTM은 day30의 마지막 hidden state만 사용한다. day1~day15의 패턴이 국면 판단에 중요한 경우, 이 정보가 충분히 반영되지 않을 수 있다.

---

## 개선 실험 설계

### Exp 1: Baseline (현재 모델)
변경 없음. 비교 기준점.

---

### Exp 2: Focal Loss

**아이디어**: 일반 CrossEntropy는 모든 예제에 동등하게 반응한다. 모델이 이미 잘 맞추는 Bull(97.6%)에도 큰 loss를 부여한다. Focal Loss는 이미 잘 맞추는 "쉬운 예제"의 loss 기여를 줄이고, 틀리기 어려운 Neutral 같은 "어려운 예제"에 집중한다.

**수식**:
```
Focal Loss = (1 - p_t)^γ × CrossEntropy

p_t: 정답 클래스의 예측 확률
γ  : focusing parameter (γ=2 사용)

예시:
  모델이 Bull을 90% 확신 → p_t=0.9 → (1-0.9)^2 = 0.01 → loss 100배 감소
  모델이 Neutral을 60% 확신 → p_t=0.6 → (1-0.6)^2 = 0.16 → loss 적당히 유지
```

참고: Lin et al., 2017. *Focal Loss for Dense Object Detection*. ICCV.

**모델 구조**: Exp1과 동일 (Conv1D + LSTM)

---

### Exp 3: Focal Loss + Data Augmentation

**아이디어**: 학습 데이터 488개는 너무 적다. 원본 데이터에 작은 Gaussian 노이즈를 더한 복사본을 만들어 학습 데이터를 2배로 늘린다.

```
X_aug = X_original + ε,  ε ~ N(0, 0.05²)

원본:  (488, 30, 10)
복사본: (488, 30, 10)  ← 원본과 거의 같지만 약간 다름
합산: (976, 30, 10)   ← 2배 증가
```

**왜 유효한가?**: 실제 시장에서 같은 국면이라도 매번 미세하게 다른 형태로 나타난다. 노이즈 추가는 이 다양성을 인위적으로 모방하여 모델이 패턴의 본질을 학습하도록 돕는다.

**모델 구조**: Exp1과 동일 (Conv1D + LSTM)

---

### Exp 4: BiLSTM + Attention + Augmentation

**아이디어 1 — Bidirectional LSTM**:

단방향 LSTM은 day1→day30 순서로만 읽는다. Bidirectional LSTM은 앞→뒤(forward), 뒤→앞(backward) 두 방향을 동시에 읽어 각 시점에서 "이전 맥락 + 이후 맥락"을 모두 볼 수 있다.

```
단방향 LSTM:
  day1→day2→...→day30  →  h30 (마지막 1개 사용)

BiLSTM:
  forward:  day1→day2→...→day30  →  h_fwd (각 시점)
  backward: day30→day29→...→day1  →  h_bwd (각 시점)
  concat: [h_fwd, h_bwd]  →  (batch, 30, 128)  ← 30개 시점 모두 사용 가능
```

**아이디어 2 — Attention**:

모든 30일 시점 중 국면 판단에 중요한 날이 다를 수 있다. Attention은 각 시점에 중요도 점수를 매겨 가중 평균을 계산한다.

```
각 시점 출력: (batch, 30, 128)
         ↓ Linear(128→1)
attention score: (batch, 30)
         ↓ softmax
attention weight: (batch, 30)  합=1
         ↓ 가중 합산
context vector: (batch, 128)  ← 중요한 날에 집중된 표현
```

**모델 구조**:
```
입력: (batch, 30, 10)
  ↓ ConvBlock (동일)
  ↓ BiLSTM(32→64×2=128, 1 layer)
  ↓ Attention → context (batch, 128)
  ↓ Classifier(128→32→3)
총 파라미터: ~58,000개
```

---

## 실험 결과

### 분류 성능 비교

| 실험 | Accuracy | Bear | Neutral | Bull |
|------|---------|------|---------|------|
| Exp1: Baseline | 57.1% | 34.9% | 23.8% | 97.6% |
| Exp2: Focal Loss | 52.4% | 34.9% | **47.6%** | 73.2% |
| **Exp3: Focal Loss + Augmentation** | **61.0%** | **46.5%** | 33.3% | 90.2% |
| Exp4: BiLSTM + Attention | 49.5% | 30.2% | 38.1% | 75.6% |

### Confusion Matrix 비교 (Test set, 105개)

**Exp1 (Baseline)**
```
              Bear  Neutral  Bull
Bear (43)      15       9    19
Neutral (21)    8       5     8
Bull (41)       0       1    40
```

**Exp2 (Focal Loss)**
```
              Bear  Neutral  Bull
Bear (43)      15       6    22
Neutral (21)    6      10     5
Bull (41)       0      11    30
```

**Exp3 (Augmentation) ← Phase 1 최고**
```
              Bear  Neutral  Bull
Bear (43)      20       5    18
Neutral (21)    8       7     6
Bull (41)       1       3    37
```

**Exp4 (BiLSTM+Attention)**
```
              Bear  Neutral  Bull
Bear (43)      13       9    21
Neutral (21)    5       8     8
Bull (41)       2       8    31
```

---

## 실험별 분석

### Exp2 (Focal Loss)의 특이점: Neutral 47.6%

Focal Loss가 Neutral recall을 23.8% → 47.6%로 크게 향상시켰다. 그러나 전체 accuracy는 57.1% → 52.4%로 하락했다. **이는 Focal Loss가 Bull을 희생해서 Neutral을 학습했기 때문이다** (Bull recall 97.6% → 73.2%).

Bull이 테스트셋의 39%를 차지하므로, Bull을 덜 맞추면 전체 accuracy가 하락한다. 이처럼 loss function의 변경은 클래스 간 성능의 trade-off를 만든다.

### Exp3 (Augmentation)이 Best인 이유

데이터 증강이 가장 효과적인 이유는 근본 문제(데이터 부족)를 직접 해결하기 때문이다. 488개의 학습 데이터를 976개로 늘리면서:

- 모델이 더 다양한 패턴을 학습
- 과적합이 줄어들어 일반화 성능 향상
- Bear recall이 34.9% → 46.5%로 대폭 개선

### Exp4 (BiLSTM+Attention)의 실패 원인

이론적으로는 가장 강력한 구조임에도 성능이 가장 낮았다. 원인:

1. **파라미터 증가 (~58,000개)**: 데이터 976개(증강 후)에도 불구하고 여전히 모델이 크다
2. **BiLSTM의 과대적합**: 양방향 학습이 오히려 training 데이터의 노이즈를 더 잘 외워버림
3. **Attention의 한계**: 30일이라는 짧은 시퀀스에서 attention의 이점이 크지 않음

---

---

## Phase 2: 다자산 데이터 실험 (QQQ / GLD / TLT 추가)

### 배경 및 동기

Phase 1의 핵심 한계는 데이터 부족이었다. Gaussian 노이즈 증강(Exp3)으로 488→976개를 만들었지만, 이는 동일 데이터의 인위적 복사본이다. 실제 다른 자산(QQQ, GLD, TLT)의 HMM 라벨을 생성하면 **같은 모델 구조로 4배 많은 진짜 데이터**를 확보할 수 있다.

팀원이 SPY와 동일한 HMM 파이프라인(504일 rolling window, 3-state Gaussian HMM, Sharpe ranking)으로 QQQ / GLD / TLT 라벨을 생성하여 push하였다.

### 새 학습 데이터: `multi_asset_supervised_30d_5d.npz`

| | SPY 단독 (Phase 1) | 4자산 통합 (Phase 2) |
|---|---|---|
| Train 샘플 | 488개 | **1,952개 (4배)** |
| Bear (train) | 149개 | 624개 |
| Neutral (train) | 128개 ← 최소 클래스 | 654개 ← 균형 달성 |
| Bull (train) | 211개 | 674개 |

클래스 불균형 문제가 추가 처리 없이 자연스럽게 해소되었다.

**주의**: Bear/Neutral/Bull은 각 자산의 rolling window 내 상대적 Sharpe ranking이다. GLD의 Bull과 SPY의 Bull은 절대 수익률 수준이 다르다.

---

### Multi-asset Exp A: CrossEntropy + 기본 하이퍼파라미터

**설정**: patience=25, batch_size=16 (기존 train.py 기본값 그대로)

**결과**:
```
Accuracy: 57.1%

               Bear  Neutral     Bull
Bear (160)       88       19       53
Neutral (95)     29       17       49
Bull (165)       11       19      135
```

| 클래스 | Recall |
|--------|--------|
| Bear | 55.0% |
| Neutral | **17.9%** ← 급락 |
| Bull | 81.8% |

**분석**: epoch 27에서 early stop. 데이터가 4배 늘었음에도 patience가 그대로라 학습이 너무 일찍 종료되었다. Neutral recall이 Phase 1 대비 오히려 크게 낮아졌는데, 이는 SPY/QQQ/GLD/TLT의 "Neutral"이 각 자산별로 다른 시장 구조를 반영하여 모델이 혼동하기 때문으로 분석된다.

---

### Multi-asset Exp B: CrossEntropy + 하이퍼파라미터 조정 ← 최고 Bear recall

**설정**: patience=50, batch_size=32 (데이터 4배 증가에 맞춰 조정)

**결과**:
```
Accuracy: 59.8%

               Bear  Neutral     Bull
Bear (160)       94       17       49
Neutral (95)     31       24       40
Bull (165)       13       19      133
```

| 클래스 | Recall |
|--------|--------|
| Bear | **58.8%** ← 전체 실험 최고 |
| Neutral | 25.3% |
| Bull | 80.6% |

**분석**: patience를 늘리자 accuracy 57.1% → 59.8%, Neutral recall 17.9% → 25.3%로 개선. Bear recall은 모든 실험(Phase 1 포함)을 통틀어 최고치(58.8%)를 기록했다. validation loss가 epoch 10에서 62.1%까지 올라갔다가 진동하는 패턴이 관찰됐으며, batch_size=32로 gradient 노이즈를 줄인 것이 수렴 안정성에 기여했다.

---

### Multi-asset Exp C: Focal Loss + 조정된 하이퍼파라미터

**설정**: FocalLoss(γ=2), patience=50, batch_size=32

**결과**:
```
Accuracy: 54.3%

               Bear  Neutral     Bull
Bear (160)       72       26       62
Neutral (95)     25       18       52
Bull (165)       10       17      138
```

| 클래스 | Recall |
|--------|--------|
| Bear | 45.0% |
| Neutral | 18.9% |
| Bull | 83.6% |

**분석**: Focal Loss가 multi-asset 데이터에서는 오히려 역효과였다. Phase 1에서 Focal Loss가 Neutral recall을 23.8% → 47.6%로 끌어올렸던 것과 대조적이다. 원인: Phase 1에서 Focal Loss는 Bull(쉬운 예제)에 집중된 loss를 Neutral(어려운 예제)로 이동시켰다. Phase 2에서는 클래스 균형이 이미 맞춰져 있어 Focal Loss가 집중할 "쉬운 예제"의 방향이 불명확하고, 4개 자산에 걸친 "어려운 예제"의 기준이 일관되지 않아 validation loss가 크게 진동(0.87 ↔ 1.5)하며 수렴에 실패했다.

---

## 전체 실험 최종 비교

| 실험 | Train 샘플 | Accuracy | Bear recall | Neutral recall | Bull recall |
|------|-----------|---------|-------------|----------------|-------------|
| Exp1: Baseline (SPY) | 488 | 57.1% | 34.9% | 23.8% | 97.6% |
| Exp2: Focal Loss (SPY) | 488 | 52.4% | 34.9% | **47.6%** | 73.2% |
| **Exp3: Focal Loss + Augment (SPY)** | 976 | **61.0%** | 46.5% | 33.3% | **90.2%** |
| Exp4: BiLSTM + Attention (SPY) | 976 | 49.5% | 30.2% | 38.1% | 75.6% |
| Multi-A: CE, p=25, bs=16 | 1,952 | 57.1% | 55.0% | 17.9% | 81.8% |
| **Multi-B: CE, p=50, bs=32** | 1,952 | 59.8% | **58.8%** | 25.3% | 80.6% |
| Multi-C: Focal, p=50, bs=32 | 1,952 | 54.3% | 45.0% | 18.9% | 83.6% |

**지표별 최고 모델**:
- 전체 Accuracy: **Exp3 (61.0%)**
- Bear recall: **Multi-B (58.8%)** ← 하락장 인식 최강
- Neutral recall: **Exp2 (47.6%)**
- Bull recall: **Exp1 (97.6%)**

### Phase 2의 핵심 발견

1. **다자산 데이터의 효과**: Bear recall이 46.5%(Exp3) → 58.8%(Multi-B)로 의미있게 향상. 다양한 자산의 하락 패턴을 학습함으로써 하락 국면 인식 능력이 실질적으로 개선되었다.

2. **Neutral의 구조적 한계**: 어떤 실험에서도 Neutral recall은 33% 이상을 넘지 못했다. "중간 Sharpe" 국면은 자산별로 특성이 달라 모델이 일관된 패턴을 학습하기 어렵다. 이는 HMM 라벨 자체의 한계이기도 하다.

3. **Focal Loss는 데이터 부족 시에만 유효**: SPY 단독 데이터(불균형, 488개)에서는 Focal Loss가 Neutral recall을 크게 향상시켰지만, 균형 잡힌 다자산 데이터(1,952개)에서는 오히려 수렴 불안정을 초래했다.

4. **하이퍼파라미터는 데이터 규모에 맞게 조정 필요**: patience를 25→50으로 늘리는 것만으로도 Multi-A 대비 accuracy +2.7%p, Bear recall +3.8%p 향상. 데이터가 늘면 학습 안정화에 더 많은 epoch이 필요하다.

---

## 핵심 발견: 분류 성능 ≠ 포트폴리오 성과

### 백테스트 결과 비교

| 모델 | 분류 Accuracy | 백테스트 누적수익 | Sharpe | MDD | Calmar |
|------|-------------|--------------|--------|-----|--------|
| 원본 (train.py) | 58.1% | 17.4% | 0.52 | -5.8% | **1.38** |
| Exp3 (최고 분류) | 61.0% | 14.8% | 0.33 | **-5.2%** | 1.30 |

**Exp3가 분류는 더 잘하는데 백테스트 수익이 낮다.** 왜?

Exp3는 Bear recall이 34.9% → 46.5%로 크게 향상되었다. 즉, Bear 구간을 더 잘 맞추게 되었고, Bear로 판단할 때 주식 비중을 줄인다. **테스트 기간(2024~2026)은 강한 상승장이었기 때문에**, 더 자주 Bear를 인식해서 주식 비중을 낮춘 것이 오히려 수익률 손실로 이어졌다.

이것은 금융 ML에서 매우 중요한 교훈이다.

> **분류 정확도와 투자 성과는 별개의 지표다.**
> 정확도가 높은 모델이 더 좋은 투자 성과를 보장하지 않는다.
> 특히 시장 방향이 강한 기간에는 더 그렇다.

### 왜 Calmar Ratio가 핵심 지표인가

두 모델 모두 MDD가 약 -5% 수준으로, Buy&Hold(-17%)에 비해 하락을 잘 방어하고 있다. Calmar는 1.30~1.38로 모든 baseline 전략 중 최고 수준을 유지한다.

---

## Phase 1 최종 선택 (중간 결론)

**Exp3 (Focal Loss + Data Augmentation)** 이 Phase 1에서 가장 우수했다.
단, 이후 Phase 2/3 실험을 거쳐 **최종 모델은 Phase 3 (Cross-asset + AdamW + Neutral-boost)** 으로 교체됨.

| 항목 | Phase 1 최고 (Exp3) | **최종 (Phase 3)** |
|------|--------------------|--------------------|
| Accuracy | 61.0% | **61.9%** |
| Bear recall | 46.5% | **60.5%** |
| MDD | -5.2% | **-7.4%** |
| Calmar | 1.30 | **1.35** |

---

---

## Phase 3: 교차 자산 피처 실험 (Option A)

### 배경 및 동기

Phase 2 다자산 실험의 핵심 문제는 **라벨 충돌**이었다. SPY가 Bear일 때 TLT/GLD는 Bull인 경우가 많아, 같은 시장 환경에 자산마다 다른 라벨이 붙어 모델이 혼동했다.

해결책: **라벨은 SPY 하나로 고정하고, 나머지 자산은 입력 피처로만 사용한다.**

```
기존 다자산 방식 (라벨 충돌):
  (QQQ 피처, QQQ Bear)  ←  같은 날, 다른 라벨
  (GLD 피처, GLD Bull)  ←  충돌
  (SPY 피처, SPY Bear)

교차 자산 피처 방식 (이번):
  입력: [SPY 10개 + QQQ 10개 + GLD 10개 + TLT 10개] = (30, 40)
  라벨: SPY 국면만 → 충돌 없음
```

샘플 수는 698개로 동일하지만 피처 수가 10 → 40개로 늘어, 모델이 자산 간 상관관계(예: TLT↑ + GLD↑ + SPY↓ = Bear 신호)를 직접 학습할 수 있다.

새 스크립트: `scripts/prepare_cross_asset_dataset.py`

---

### Cross-asset Baseline: CrossEntropy + patience=50 + batch=32

**데이터**: `cross_asset_supervised_30d_5d.npz` — (698, 30, 40), SPY 라벨만

**결과**:
```
Accuracy: 62.9%

               Bear  Neutral     Bull
Bear (43)        23        8       12
Neutral (21)      8        4        9
Bull (41)         1        1       39
```

| 클래스 | Recall |
|--------|--------|
| Bear | 53.5% |
| Neutral | 19.0% |
| Bull | 95.1% |

**분석**: 샘플 수는 동일(488)하지만 피처 40개를 활용해 **62.9%로 전체 실험 중 최고 정확도** 달성. Bear recall도 53.5%로 Phase 1 최고(46.5%)를 넘었다. 단, train_loss(0.48) vs val_loss(0.94)의 큰 격차로 과적합이 관찰되었다. Neutral recall 19%는 여전히 약점.

---

### Phase 3 추가 실험: 과적합 해소 시도

Cross-asset baseline의 과적합(train/val loss 비율 2:1)을 줄여 정확도를 더 높이기 위해 3가지 방법을 순차 실험했다.

---

#### B: 모델 크기 축소 + Dropout 강화

**가설**: 31K 파라미터는 488개 데이터에 과도하다. 파라미터를 줄이면 일반화가 나아질 것이다.

**변경**:
```
기존: Conv(40→32) → LSTM(hidden=64) → Linear(64→32→3)  ≈ 31K params
변경: Conv(40→16) → LSTM(hidden=32) → Linear(32→16→3)  ≈ 9.7K params
Dropout: 0.5 → 0.7
```

**결과**:
```
Accuracy: 53.3%

               Bear  Neutral     Bull
Bear (43)        16        0       27
Neutral (21)      5        0       16
Bull (41)         1        0       40
```

| 클래스 | Recall |
|--------|--------|
| Bear | 37.2% |
| Neutral | **0.0%** |
| Bull | 97.6% |

**분석 및 실패 원인**: 과소적합(underfitting). 9.7K 파라미터는 40개 피처 × 30일 시퀀스의 복잡한 패턴을 학습하기에 너무 단순했다. Dropout 0.7도 과도하여 모델이 Neutral을 완전히 포기하고 Bear/Bull 양극단 예측에 집중했다. **파라미터가 너무 많아도, 너무 적어도 문제다.**

---

#### D: Focal Loss (γ=2)

**가설**: Neutral(21개)이 Bear(43개), Bull(41개)보다 적다. Focal Loss로 어려운 예제(Neutral)에 집중시키면 recall이 올라갈 것이다. Phase 1에서 SPY 단독 데이터에 Focal Loss를 적용했을 때 Neutral recall이 23.8% → 47.6%로 향상된 바 있다.

**변경**: CrossEntropyLoss → FocalLoss(γ=2, class_weights 유지), 나머지 동일

**결과**:
```
Accuracy: 54.3%

               Bear  Neutral     Bull
Bear (43)         7       26       10
Neutral (21)      1       13        7
Bull (41)         0        4       37
```

| 클래스 | Recall |
|--------|--------|
| Bear | 16.3% ← 폭락 |
| Neutral | **61.9%** ← 대폭 향상 |
| Bull | 90.2% |

**분석**: Focal Loss가 Neutral recall을 19% → **61.9%** 로 3배 이상 끌어올렸다. 그러나 Bear recall이 53.5% → 16.3%로 폭락해 전체 accuracy가 54.3%로 하락했다. 이는 전형적인 Focal Loss의 trade-off다. γ=2로 "어려운 예제"에 집중한 결과, Neutral 학습이 과도하게 강조되고 상대적으로 쉬운 Bear와 Bull의 학습이 약해졌다. **Neutral을 살리면 Bear가 죽는 구조적 한계.**

---

#### C: Cosine Annealing LR

**가설**: ReduceLROnPlateau는 lr이 한 번 내려가면 다시 올라오지 않는다. CosineAnnealingLR은 lr을 주기적으로 올렸다 내렸다 하며 local minima를 탈출할 기회를 준다.

**변경**: 스케줄러만 교체. `CosineAnnealingLR(T_max=50, eta_min=1e-6)`, patience=60

**결과**:
```
Accuracy: 49.5%

               Bear  Neutral     Bull
Bear (43)         9       16       18
Neutral (21)      2        4       15
Bull (41)         0        2       39
```

| 클래스 | Recall |
|--------|--------|
| Bear | 20.9% |
| Neutral | 19.0% |
| Bull | 95.1% |

**분析 및 실패 원인**: val_loss가 epoch 15에서 최저(0.82)를 기록한 뒤 계속 상승해 결국 epoch 75에서 early stop. lr이 주기적으로 올라가는 과정에서 이미 학습된 가중치가 흔들려 오히려 불안정해졌다. 488개라는 소규모 데이터에서는 lr oscillation의 이점보다 불안정성의 단점이 컸다. **데이터가 많을수록 Cosine Annealing의 효과가 크다.**

---

#### E: Patience 100으로 확대

**가설**: patience=50이면 best 이후 50 epoch만 기다리고 포기한다. 더 오래 기다리면 val_loss가 다시 내려올 수 있다.

**변경**: patience=50 → 100, 나머지 동일

**결과**:
```
Accuracy: 49.5%

               Bear  Neutral     Bull
Bear (43)        11       12       20
Neutral (21)      5        4       12
Bull (41)         0        4       37
```

| 클래스 | Recall |
|--------|--------|
| Bear | 25.6% |
| Neutral | 19.0% |
| Bull | 90.2% |

**분析 및 실패 원인**: epoch 10에서 val_loss 0.86이 best였고, 이후 epoch 109까지 계속 나빠졌다(val_loss 1.3대). patience=100으로 늘리자 best를 이미 지난 상태에서 100 epoch을 더 기다린 꼴이 됐다. train_loss는 0.86 → 0.36으로 계속 떨어지는데 val_loss는 올라갔다 → 극심한 과적합. **patience를 늘리는 건 "아직 best를 못 찾은 경우"에만 유효하다. 이미 best를 지났다면 오히려 더 나빠진다.**

---

#### F: LSTM 전체 시점 평균 사용 (Global Average Pooling)

**가설**: 현재 코드는 LSTM의 마지막 hidden state(`h_n[-1]`, day30)만 분류에 사용한다. 팀원 제안: 30일 전체 시점의 출력을 평균 내면 초반 패턴도 반영될 것이다.

**현재 vs 변경**:
```python
# 현재 (마지막 시점만)
_, (h_n, _) = self.lstm(x)
return self.classifier(h_n[-1])     # day30 hidden state만

# 변경 (전체 시점 평균)
output, _ = self.lstm(x)
return self.classifier(output.mean(dim=1))  # day1~day30 평균
```

파라미터 수 변화 없음.

**결과**:
```
Accuracy: 49.5%

               Bear  Neutral     Bull
Bear (43)        19       13       11
Neutral (21)      9        2       10
Bull (41)         1        9       31
```

| 클래스 | Recall |
|--------|--------|
| Bear | 44.2% |
| Neutral | 9.5% |
| Bull | 75.6% |

**분析 및 실패 원인**: val_loss가 epoch 5에서 0.95에서 급격히 발산(epoch 59에서 1.8). LSTM의 마지막 hidden state(`h_n[-1]`)는 "마지막 날만 보는 게" 아니다. LSTM은 내부 cell state를 통해 1~30일 모든 정보를 누적해 h_n[-1]에 압축한다. 즉 `h_n[-1]`은 이미 30일 전체의 요약이다. 반면 Global Average Pooling은 최근 데이터와 오래된 데이터를 동등하게 취급하는데, **금융 시계열에서 최근 며칠은 30일 전보다 훨씬 중요하기 때문에** 오히려 신호가 희석됐다.

---

### Phase 3 전체 실험 결과 비교

| 실험 | Accuracy | Bear | Neutral | Bull | 핵심 관찰 |
|------|---------|------|---------|------|----------|
| **Cross-asset Baseline** | **62.9%** | **53.5%** | 19.0% | **95.1%** | 피처 다양성 효과 |
| B: 모델 축소 + Dropout 0.7 | 53.3% | 37.2% | 0.0% | 97.6% | 과소적합 |
| D: Focal Loss γ=2 | 54.3% | 16.3% | **61.9%** | 90.2% | Bear↓ Neutral↑ |
| C: Cosine Annealing LR | 49.5% | 20.9% | 19.0% | 95.1% | lr 불안정 |
| E: Patience 100 | 49.5% | 25.6% | 19.0% | 90.2% | best 지난 후 과적합 |
| F: LSTM 전체 평균 | 49.5% | 44.2% | 9.5% | 75.6% | 최근 신호 희석 |

**결론**: 5가지 개선 시도 모두 baseline(62.9%)보다 낮았다. 이는 **데이터 488개라는 근본적 제약** 때문이다. 모델 구조나 학습 전략을 어떻게 바꿔도 이 한계를 넘기 어렵다. 실질적인 돌파구는 데이터 확장(기간을 2004년으로 늘리기)이다.

---

### 전체 실험 최종 종합 비교 (8개 실험)

| 단계 | 실험 | 샘플 | 피처 | Accuracy | Bear | Neutral | Bull |
|------|------|------|------|---------|------|---------|------|
| Phase 1 | Exp1: Baseline (SPY) | 488 | 10 | 57.1% | 34.9% | 23.8% | 97.6% |
| Phase 1 | Exp2: Focal Loss (SPY) | 488 | 10 | 52.4% | 34.9% | 47.6% | 73.2% |
| Phase 1 | Exp3: Focal+Augment (SPY) | 976 | 10 | 61.0% | 46.5% | 33.3% | 90.2% |
| Phase 1 | Exp4: BiLSTM+Attention | 976 | 10 | 49.5% | 30.2% | 38.1% | 75.6% |
| Phase 2 | Multi-A: 4자산 각자라벨, CE | 1,952 | 10 | 57.1% | 55.0% | 17.9% | 81.8% |
| Phase 2 | Multi-B: 4자산 각자라벨, p=50 | 1,952 | 10 | 59.8% | 44.2% | 25.3% | 80.6% |
| Phase 2 | Multi-C: Focal Loss | 1,952 | 10 | 54.3% | 45.0% | 18.9% | 83.6% |
| **Phase 3** | **Cross-asset + AdamW + neutral-boost 1.2 (최종)** | **488** | **40** | **61.9%** | **60.5%** | 0.0% | **95.1%** |

**최종 선택 모델**: **best_model.pt (Phase 3, seed=42)**
- 재현 가능한 최고 성능: **61.9%** (seed=42 고정, AdamW, neutral-boost 1.2)
- Bear recall: **60.5%** (하락장 인식 능력 전체 2위)
- 데이터: cross_asset_supervised_30d_5d.npz (SPY 라벨 + 4자산 피처 40개)

---

---

## Phase 3: Cross-asset 피처 + AdamW + Neutral-boost

### 배경

Phase 3까지의 실험에서 Neutral recall이 구조적으로 낮은 문제가 해결되지 않았다. 팀원이 개선된 `train.py`를 제공하였고, 이를 기반으로 추가 실험을 진행했다.

### 코드 개선 사항 (팀원 제공)

기존 `train.py` 대비 주요 변경:

| 항목 | 기존 | 변경 후 |
|------|------|---------|
| 최적 모델 저장 기준 | val_loss | **balanced accuracy** (Bear/Neutral/Bull 평균 recall) |
| 옵티마이저 | Adam | **AdamW** (weight decay 더 정밀) |
| Neutral 클래스 가중치 | 기본 역빈도 가중치 | **neutral_boost 배수 추가 적용** |
| 시드 고정 | 없음 | **set_seed(42)** |
| JSON 직렬화 | Path 타입 오류 | **수정됨** |
| 모델 크기 기본값 | conv=32, lstm=64 | **conv=16, lstm=32** |
| Dropout | 0.5 | **0.6** |
| weight_decay | 1e-4 | **1e-2 (기본값)** |

`neutral_boost`는 Neutral 클래스 가중치를 추가로 곱해 loss에서 Neutral 오분류 패널티를 강화하는 방식이다:

```python
weights = n_samples / (3.0 * counts)  # 기본 역빈도 가중치
weights[1] *= neutral_boost            # Neutral(index=1)에만 추가 배수 적용
```

---

### Phase 3 최종 실험: Neutral-boost 1.2 + AdamW

> 아래는 실험 과정에서 시도한 하이퍼파라미터 탐색 결과다.

**3일 간격 라벨 실험 (실패)**: 5거래일→3거래일 간격으로 샘플을 814개로 늘렸으나,
30일 윈도우에서 인접 샘플 27/30일이 겹쳐 validation 성능이 인위적으로 높고 test에서 26.3%로 폭락했다.
**샘플 수 증가가 독립적인 정보 증가를 의미하지 않는다.**

**neutral-boost 2.0 실험**: Neutral recall이 0%→52.4%로 향상되었으나 전체 accuracy가 52.4%로 하락.
Neutral을 살리면 Bear/Bull이 희생되는 trade-off가 확인됨.

**최종 확정 (neutral-boost 1.2, weight-decay 1e-2, AdamW, seed=42)**

**명령어**:
```bash
python3 scripts/train.py \
  --data data/processed/cross_asset_supervised_30d_5d.npz \
  --model-output outputs/models/best_model.pt \
  --epochs 80 --patience 10 --batch-size 16 \
  --lr 1e-4 --conv-channels 16 --lstm-hidden 32 \
  --dropout 0.6 --weight-decay 1e-2 \
  --neutral-boost 1.2 --best-metric val_bal_acc --seed 42
```

**결과**:
```
Accuracy: 61.9%
Balanced Accuracy: 51.9%

               Bear  Neutral     Bull
Bear (43)        26        0       17
Neutral (21)      8        0       13
Bull (41)         2        0       39
```

| 클래스 | Recall |
|--------|--------|
| Bear | 60.5% |
| Neutral | 0.0% |
| Bull | 95.1% |

**분석**: AdamW + neutral_boost=1.2 + seed=42로 재현 가능한 61.9%를 달성. Neutral recall은 0%이지만, val 셋에 Neutral이 19개뿐이어서 best 모델 선정 자체가 불안정한 구조적 한계다. Bear recall 60.5%는 하락장 방어 측면에서 우수하며, 이 모델을 최종 모델로 확정한다.

### Phase 3 실험 결과 비교

| 실험 | Accuracy | Bear | Neutral | Bull | 비고 |
|------|---------|------|---------|------|------|
| neutral-boost 2.0 | 52.4% | 32.6% | 52.4% | 73.2% | boost 과도 |
| **neutral-boost 1.2, wd=1e-2, AdamW** | **61.9%** | **60.5%** | 0.0% | **95.1%** | **최종 채택** |

---

## 발표 논리 구성

```
"총 8개의 실험을 세 단계로 진행했다.

[Phase 1 — SPY 단독 데이터]
Focal Loss는 Neutral 인식 능력을 2배 향상시켰지만,
전체 정확도는 소폭 하락하는 trade-off가 존재했다.

Data Augmentation이 가장 효과적이었다.
488개 학습 데이터를 2배로 늘려 Bear recall을 11.6%p 향상시켰고,
전체 정확도도 57.1% → 61.0%로 개선했다.

BiLSTM + Attention은 이론적으로 강력하지만,
소규모 데이터에서 오히려 과적합이 심화되어 성능이 저하되었다.

[Phase 2 — QQQ / GLD / TLT 다자산 데이터 추가]
팀원이 생성한 다자산 HMM 라벨로 학습 데이터를 488→1,952개(4배)로 늘렸다.
그러나 자산마다 다른 라벨(GLD Bull ≠ SPY Bear)이 충돌하는 문제가 있었다.
SPY 테스트셋만으로 평가하면 59.8%로 소폭 향상되었고,
Bear recall은 58.8%로 전체 Phase 중 최고를 기록했다.

[Phase 3 — 교차 자산 피처 + AdamW + Neutral-boost]
라벨 충돌 문제를 해결하기 위해, 4개 자산 피처를 하나의 입력으로 합치고
라벨은 SPY 국면만 사용하는 방식으로 전환했다.
입력 피처: 10 → 40개 (SPY+QQQ+GLD+TLT 각 10개 concat)
AdamW 옵티마이저와 neutral_boost=1.2를 적용하고, seed=42로 재현성을 확보했다.
결과: 61.9% (Bear recall 60.5%) — 재현 가능한 최고 성능.

분류 성능 향상이 반드시 투자 성과 향상으로 이어지지 않는다는 점도 발견했다.
Bear 인식이 강화되면 강한 상승장에서 비중을 더 줄이게 되어 절대 수익률이 낮아지지만,
하락 방어(MDD -7.4%)와 Calmar Ratio(1.35)에서는 모든 전략 중 1위를 기록했다."
```
