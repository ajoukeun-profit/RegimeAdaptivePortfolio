# Binary Soft Regime-MVO 수학적 해석

> Notion 붙여넣기용 요약입니다. 현재 최종 실험인 `Binary Soft Label + 2-Regime MVO + cap 40%` 기준으로 정리했습니다.

## 1. 연구 목적

본 프로젝트의 목적은 단순히 시장 국면을 분류하는 것이 아니라, 예측된 국면 확률을 이용해 포트폴리오 비중을 동적으로 조절하는 것이다.

최종 구조는 다음과 같다.

```text
HMM pseudo-label
-> Binary soft-label classifier
-> 2-Regime MVO
-> Weight cap 40%
-> Portfolio backtest
```

분류 성능은 중간 지표이고, 최종 평가는 포트폴리오 성과로 판단한다.

---

## 2. HMM Label은 True Label이 아니라 Pseudo-label

HMM은 관측 가능한 시장 데이터로부터 직접 관측되지 않는 잠재 상태를 추정한다.

$$
s_t \in \{\text{Bear}, \text{Neutral}, \text{Bull}\}
$$

하지만 이 상태는 사람이 직접 부여한 절대적 정답이 아니다. HMM이 수익률, 변동성, Sharpe 특성 등을 바탕으로 만든 통계적 pseudo-label이다.

따라서 딥러닝 모델의 학습 목표는 다음과 같이 이해한다.

$$
f_{\theta}(X_t) \approx \hat{y}_{t+h}^{HMM}
$$

여기서 \(X_t\)는 최근 30거래일의 cross-asset feature이고, \(\hat{y}_{t+h}^{HMM}\)는 HMM이 만든 미래 horizon의 pseudo-label이다.

---

## 3. 왜 3-class에서 Binary로 바꿨는가

초기 모델은 다음 세 상태를 예측했다.

$$
\{\text{Bear}, \text{Neutral}, \text{Bull}\}
$$

하지만 test set에서 Neutral recall이 0.0%로 나타났다. Neutral은 HMM의 중간 상태라 경제적 경계가 불명확하고, Bear 또는 Bull과 겹치는 구간이 많다.

투자 목적은 모든 국면을 세밀하게 구분하는 것이 아니라 하방 위험을 인식하는 것이다. 따라서 최종 task는 다음 binary 문제로 재정의했다.

$$
y^{bin}_t =
\begin{cases}
1, & \text{Bear} \\
0, & \text{Non-Bear}
\end{cases}
$$

여기서

$$
\text{Non-Bear} = \text{Neutral} \cup \text{Bull}
$$

이다.

---

## 4. Binary Soft Label

Hard label은 한 샘플을 하나의 정답으로만 취급한다.

```text
Bear = [0, 1]
Non-Bear = [1, 0]
```

하지만 HMM은 각 상태의 posterior probability를 제공한다. 이를 binary로 합치면 다음 soft target을 만들 수 있다.

$$
q_t =
\begin{bmatrix}
P(\text{Non-Bear} \mid X_t) \\
P(\text{Bear} \mid X_t)
\end{bmatrix}
$$

그리고

$$
P(\text{Non-Bear}) =
P(\text{Neutral}) + P(\text{Bull})
$$

이다.

모델 출력은 다음과 같다.

$$
p_{\theta}(X_t) =
\begin{bmatrix}
P_{\theta}(\text{Non-Bear} \mid X_t) \\
P_{\theta}(\text{Bear} \mid X_t)
\end{bmatrix}
$$

---

## 5. Soft Cross Entropy Loss

Hard label cross entropy는 정답 클래스 하나에 대해서만 loss를 계산한다. Soft label에서는 target이 확률분포이므로 전체 분포 차이를 줄이는 방향으로 학습한다.

$$
\mathcal{L}_t(\theta)
=
-
\sum_{k \in \{\text{Non-Bear}, \text{Bear}\}}
q_{t,k}
\log p_{\theta,k}(X_t)
$$

전체 학습 loss는 다음과 같다.

$$
\mathcal{L}(\theta)
=
\frac{1}{N}
\sum_{t=1}^{N}
\mathcal{L}_t(\theta)
$$

이 방식은 HMM label의 불확실성을 완전히 버리지 않는다는 장점이 있다.

---

## 6. 최적화 방법

모델 파라미터 \(\theta\)는 soft cross entropy를 최소화하도록 학습된다.

$$
\theta^*
=
\arg\min_{\theta}
\mathcal{L}(\theta)
$$

최적화는 AdamW를 사용한다.

$$
\theta_{t+1}
=
\theta_t
- \alpha \cdot \text{AdamUpdate}
\left(
\nabla_{\theta}\mathcal{L}(\theta_t)
\right)
- \alpha \lambda \theta_t
$$

여기서 \(\alpha\)는 learning rate이고, \(\lambda\)는 weight decay이다. 과적합과 불안정한 gradient를 줄이기 위해 dropout, early stopping, gradient clipping을 함께 사용한다.

---

## 7. 2-Regime MVO

MVO는 기대수익률과 공분산을 이용해 포트폴리오 비중을 정한다. 기본 아이디어는 위험 대비 수익, 즉 Sharpe Ratio를 최대화하는 것이다.

자산 비중 벡터를 \(w\), 기대수익률 벡터를 \(\mu\), 공분산 행렬을 \(\Sigma\)라고 하면 포트폴리오 수익률과 변동성은 다음과 같다.

$$
\mu_p = w^\top \mu
$$

$$
\sigma_p = \sqrt{w^\top \Sigma w}
$$

Sharpe-max MVO는 다음 문제로 쓸 수 있다.

$$
w^*
=
\arg\max_w
\frac{w^\top \mu}{\sqrt{w^\top \Sigma w}}
$$

제약조건은 다음과 같다.

$$
\sum_i w_i = 1
$$

$$
0 \leq w_i \leq c
$$

최종 실험에서는

$$
c = 0.4
$$

를 사용했다.

---

## 8. Binary 확률과 MVO 비중 결합

훈련 구간에서 두 개의 MVO 비중을 계산한다.

$$
w_{\text{Non-Bear}}^{MVO}
$$

$$
w_{\text{Bear}}^{MVO}
$$

테스트 시점 \(t\)에서 딥러닝 모델이 예측한 확률을 사용해 최종 비중을 만든다.

$$
w_t
=
P_{\theta}(\text{Non-Bear} \mid X_t)
\cdot
w_{\text{Non-Bear}}^{MVO}
+
P_{\theta}(\text{Bear} \mid X_t)
\cdot
w_{\text{Bear}}^{MVO}
$$

즉, 모델은 직접 자산 비중을 출력하지 않는다. 모델은 국면 확률을 출력하고, 포트폴리오 비중은 MVO를 통해 계산된 regime-level weight를 확률 가중 평균하여 만든다.

---

## 9. 왜 MVO Cap이 필요한가

MVO는 평균과 공분산 추정에 민감하다. 특히 regime별로 데이터를 나누면 각 regime의 샘플 수가 작아지고, 이때 MVO는 특정 자산에 극단적으로 몰리는 비중을 선택하기 쉽다.

제약 없는 binary MVO는 다음과 같은 extreme allocation을 만들었다.

| Cap | Non-Bear MVO | Bear MVO |
|---|---|---|
| None | SPY 100% | TLT 100% |
| 40% | SPY 40%, QQQ 40%, GLD 20% | GLD 40%, TLT 40%, SPY 20% |

따라서 cap은 단순한 장식이 아니라, 작은 샘플에서 발생하는 MVO 추정 오차와 자산 쏠림을 줄이는 안정화 장치이다.

---

## 10. 최종 결과 해석

최종 비교 결과는 다음과 같다.

| Strategy | CumRet | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

해석은 다음과 같다.

1. Neutral label은 구조적으로 애매했고, binary Bear detection이 더 안정적이었다.
2. Binary soft label은 HMM posterior의 불확실성을 보존해 Bear recall을 개선했다.
3. MVO cap은 작은 샘플에서 발생하는 extreme weight 문제를 완화했다.
4. 최종적으로 Binary Soft Label + 2-Regime MVO + cap 40%가 현재 비교군 중 가장 균형 잡힌 risk-return 결과를 냈다.

단, cap 40%는 현재 실험 구간에서의 최선 결과이므로, 엄밀한 결론을 위해서는 validation 또는 walk-forward 방식으로 cap을 선택해야 한다.
