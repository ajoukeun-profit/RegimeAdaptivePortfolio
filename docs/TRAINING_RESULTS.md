# 모델 학습 결과 분석

> Step 4 완료 기준 문서  
> 발표 및 보고서 활용용

---

## 최종 모델 구성

### Architecture

```
입력: (batch, 30, 10)
        ↓
[ConvBlock]
  Conv1d(10→32, kernel=3, same padding)
  BatchNorm1d + ReLU
  Conv1d(32→32, kernel=3, same padding)
  BatchNorm1d + ReLU
        ↓ (batch, 30, 32)
[LSTM]
  hidden_size=64, num_layers=1, batch_first=True
        ↓ 마지막 hidden state
(batch, 64)
        ↓
[Classifier]
  Linear(64→32) + ReLU + Dropout(0.5)
  Linear(32→3)
        ↓
(batch, 3)  →  softmax → [p_bear, p_neutral, p_bull]
```

### 파라미터 수

| 파트 | 파라미터 수 |
|------|------------|
| ConvBlock | ~4,300 |
| LSTM | ~24,800 |
| Classifier | ~2,200 |
| **Total** | **~31,300** |

---

## 학습 실험 과정

### 실험 1: 초기 모델 (과대적합)

| 설정 | 값 |
|------|----|
| conv_channels | 64 |
| lstm_hidden | 128, 2 layers |
| dropout | 0.3 |
| batch_size | 32 |
| lr | 1e-3 |

**결과:**

| 지표 | 값 |
|------|----|
| Early Stop | epoch 17 |
| Test Accuracy | **51.4%** |

**Confusion Matrix:**

```
              Bear  Neutral  Bull
Bear (43)      14       6    23
Neutral (21)    6       0    15
Bull (41)       0       1    40
```

**문제점 진단:**

1. **심각한 과대적합 (Overfitting)**
   - 파라미터 수: 247,267개 vs 학습 데이터 488개 → 비율 507:1
   - train_loss는 epoch마다 꾸준히 감소, val_loss는 epoch 3 이후 폭등
   - Best model이 epoch 2~3 수준에서 결정됨 → 학습이 거의 안 된 상태

2. **Neutral 클래스 완전 무시 (0/21)**
   - 모델이 "항상 Bull 예측" 전략을 학습해버림
   - Bull이 211개(43.2%)로 가장 많아서 Bull만 찍어도 loss가 낮기 때문

---

### 실험 2: 모델 축소

| 변경 사항 | 이전 → 이후 | 이유 |
|-----------|------------|------|
| conv_channels | 64 → 32 | 파라미터 수 감소 |
| lstm_hidden | 128 → 64 | 파라미터 수 감소 |
| lstm_layers | 2 → 1 | 과적합 감소 |
| dropout | 0.3 → 0.5 | 정규화 강화 |
| patience | 15 → 25 | 더 오래 학습 허용 |
| weight_decay | 없음 → 1e-4 | L2 정규화 추가 |

**결과:**

| 지표 | 값 |
|------|----|
| Early Stop | epoch 32 |
| Test Accuracy | **57.1%** (+5.7%p) |

```
              Bear  Neutral  Bull
Bear (43)      14      10    19
Neutral (21)    7       6     8
Bull (41)       0       1    40
```

Neutral이 0→6으로 개선. 모델이 Neutral을 인식하기 시작.

---

### 실험 3: Label Smoothing + 배치 크기 조정 (최종)

| 변경 사항 | 이전 → 이후 | 이유 |
|-----------|------------|------|
| batch_size | 32 → 16 | 488개 소규모 데이터에서 업데이트 횟수 증가 |
| lr | 1e-3 → 3e-4 | 작은 배치에 맞게 lr 감소 |
| label_smoothing | 없음 → 0.1 | 모델 과신 방지, 일반화 향상 |

**Label Smoothing이란?**
- 일반 CrossEntropy: 정답=1.0, 오답=0.0 (one-hot)
- Label Smoothing: 정답=0.9, 오답=각 0.05 (부드러운 타겟)
- 모델이 특정 클래스를 100% 확신하는 걸 억제 → 경계 사례에서 더 좋은 일반화

**최종 결과:**

| 지표 | 값 |
|------|----|
| Early Stop | epoch 31 |
| Test Accuracy | **58.1%** |

```
              Bear  Neutral  Bull
Bear (43)      16      10    17
Neutral (21)    8       5     8
Bull (41)       1       0    40
```

---

## 최종 성능 비교

### Accuracy

| 전략 | Test Accuracy | 비고 |
|------|--------------|------|
| Random Guess | 33.3% | 3클래스 균등 확률 |
| Majority Class (항상 Bull) | 39.0% | 가장 단순한 baseline |
| **Conv1D + LSTM (최종)** | **58.1%** | **+19.1%p vs random** |

### 클래스별 정확도 (최종 모델)

| 국면 | 맞춘 수 | 전체 | 정확도 |
|------|--------|------|--------|
| Bear | 16 | 43 | **37.2%** |
| Neutral | 5 | 21 | **23.8%** |
| Bull | 40 | 41 | **97.6%** |

---

## 성능 한계와 원인 분석

### 1. 데이터 부족

학습 데이터가 488개로 금융 딥러닝에서 매우 소규모다. 5거래일 간격 라벨링이어서 14년치 데이터가 698개 샘플로 압축된 결과다.

```
14년 × 252거래일 = 3,528일
5거래일 간격 → 3,528 / 5 = 705개 샘플
train 70% = 488개
```

### 2. 분포 이동 (Distribution Shift)

```
Train 기간  (~2020): 안정적 상승장 포함
Valid 기간  (2020~2022): COVID 급락 포함
Test 기간   (2022~2026): 금리 인상, AI 붐 포함
```

각 기간마다 시장 특성이 달라 train에서 학습한 패턴이 test에서 그대로 적용되기 어렵다.

### 3. Neutral 클래스의 구조적 어려움

Neutral은 "Bull도 Bear도 아닌 상태"로 명확한 특징이 없다. HMM 라벨 기준으로도 Neutral은 "상대적으로 중간 Sharpe" 이기 때문에 애매한 구간이다.

### 4. 이 정확도가 의미하는 것

금융 시계열 분류에서 58%는 나쁜 수치가 아니다. 중요한 건 **분류 정확도가 아니라 포트폴리오 성과**다. 모델이 Bear를 Bear로 맞추는 것보다, **Bull 때 주식 비중을 높이고 Bear 때 낮추는 경향**이 있으면 충분히 수익을 낼 수 있다.

---

## 다음 단계

Step 5에서 이 모델의 확률 출력을 포트폴리오 비중으로 변환하고, Buy&Hold / 60-40 등 baseline과 Sharpe Ratio, MDD를 비교한다. 분류 정확도보다 이 백테스트 결과가 최종 평가의 핵심이다.
