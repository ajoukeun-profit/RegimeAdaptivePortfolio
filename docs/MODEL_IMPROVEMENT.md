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

**Exp3 (Augmentation) ← 최종 선택**
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

## 최종 모델 선택

**Exp3 (Focal Loss + Data Augmentation)** 를 최종 모델로 선택한다.

**이유**:
1. 전체 분류 정확도 61.0% (4개 실험 중 최고)
2. Bear recall 46.5% (4개 실험 중 최고) — 하락장 인식 능력 향상
3. 포트폴리오 MDD -5.2% (전체 전략 중 최저) — 하락 방어 유지
4. Calmar Ratio 1.30 — 전체 전략 상위권 유지

---

## 발표 논리 구성

```
"총 4개의 실험을 통해 모델을 개선했다.

Focal Loss는 Neutral 인식 능력을 2배 향상시켰지만,
전체 정확도는 소폭 하락하는 trade-off가 존재했다.

Data Augmentation이 가장 효과적이었다.
488개 학습 데이터를 2배로 늘려 Bear recall을 11.6%p 향상시켰고,
전체 정확도도 57.1% → 61.0%로 개선했다.

BiLSTM + Attention은 이론적으로 강력하지만,
소규모 데이터에서 오히려 과적합이 심화되어 성능이 저하되었다.
이는 데이터 크기와 모델 복잡도 간의 균형의 중요성을 보여준다.

분류 성능 향상이 반드시 투자 성과 향상으로 이어지지 않는다는 점도 발견했다.
정확도가 더 높은 Exp3가 백테스트 수익률은 낮았는데,
이는 강한 상승장에서 더 정확한 Bear 인식이 오히려 비중 축소로 이어졌기 때문이다."
```
