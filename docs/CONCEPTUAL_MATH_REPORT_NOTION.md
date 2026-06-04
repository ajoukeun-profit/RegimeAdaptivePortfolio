# HMM 기반 시장 국면 예측과 포트폴리오 리밸런싱의 수학적 해석

> Notion 붙여넣기용 버전입니다. 수식 블록은 Notion에서 인식하기 쉬운 `$$ ... $$` 형식으로 정리했습니다.

## 1. 연구 목적

본 프로젝트의 목적은 단순히 시장 국면(Bear, Neutral, Bull)을 분류하는 데 있지 않다. 최종 목적은 예측된 시장 국면 정보를 이용하여 포트폴리오를 동적으로 리밸런싱하고, 그 전략이 정적 포트폴리오 대비 경제적으로 유의미한 성과 개선을 보이는지 검증하는 것이다.

따라서 본 프로젝트는 다음과 같은 2단계 구조를 가진다.

1. HMM으로 생성한 시장 국면 pseudo-label을 딥러닝 모델이 예측한다.
2. 예측된 국면 확률을 포트폴리오 비중으로 변환하여 백테스트 성과를 평가한다.

중요한 점은 이 두 단계가 완전히 독립적인 task가 아니라는 것이다. HMM state는 절대적인 정답 label이 아니라 통계적 모델이 생성한 pseudo-label이므로, 해당 state를 잘 예측했다는 사실만으로 시장 예측 또는 경제적 수익성을 주장할 수 없다. 따라서 분류 성능은 중간 평가 지표이며, 최종 검증은 포트폴리오 백테스트를 통해 이루어져야 한다.

---

## 2. HMM State의 의미: True Label이 아닌 Pseudo-label

HMM(Hidden Markov Model)은 관측 가능한 수익률 데이터로부터 직접 관측되지 않는 잠재 상태(hidden state)를 추정하는 확률 모형이다.

시간 $t$에서의 관측값을 $x_t$, 잠재 시장 상태를 $s_t$라고 하자.

$$
s_t \in \{1,2,3\}
$$

본 프로젝트에서는 HMM이 추정한 세 상태를 사후적으로 Bear, Neutral, Bull로 매핑하였다. 즉,

$$
s_t \in \{\text{Bear}, \text{Neutral}, \text{Bull}\}
$$

하지만 이 label은 사람이 직접 정의한 절대적 정답이 아니다. HMM이 과거 수익률, 변동성, Sharpe 특성 등을 기반으로 추정한 통계적 상태이다. 따라서 HMM label은 다음과 같이 이해해야 한다.

$$
\hat{y}_t^{HMM} = g_{\phi}(x_{1:t})
$$

여기서 $g_{\phi}$는 HMM 기반 label 생성 함수이고, $\hat{y}_t^{HMM}$은 실제 정답 $y_t$가 아니라 pseudo-label이다.

따라서 딥러닝 모델이 학습하는 대상은 다음과 같다.

$$
f_{\theta}(X_t) \approx \hat{y}_{t+h}^{HMM}
$$

즉, 모델은 미래의 절대적 시장 상태를 직접 예측하는 것이 아니라, HMM이 정의한 시장 국면 체계를 근사하도록 학습한다.

---

## 3. 딥러닝 모델의 분류 문제

딥러닝 모델의 입력은 최근 30일간의 시장 피처이다.

$$
X_i \in \mathbb{R}^{30 \times d}
$$

최종 cross-asset 모델에서는 4개 자산(SPY, QQQ, GLD, TLT)의 피처를 결합하므로,

$$
d = 40
$$

모델은 Conv1D와 LSTM을 통해 입력 시계열을 압축하고, 세 개의 logit을 출력한다.

$$
z_i = f_{\theta}(X_i)
     =
\begin{bmatrix}
z_{i,\text{Bear}} \\
z_{i,\text{Neutral}} \\
z_{i,\text{Bull}}
\end{bmatrix}
$$

이 logit은 softmax 함수를 통해 확률로 변환된다.

$$
p_{i,k}
= P_{\theta}(y_i=k \mid X_i)
= \frac{\exp(z_{i,k})}{\sum_{j=1}^{3}\exp(z_{i,j})}
$$

따라서 모델의 출력은 다음과 같은 국면 확률 벡터이다.

$$
p_i =
\begin{bmatrix}
P(\text{Bear} \mid X_i) \\
P(\text{Neutral} \mid X_i) \\
P(\text{Bull} \mid X_i)
\end{bmatrix}
$$

---

## 4. 손실함수: Weighted Cross Entropy

본 프로젝트는 회귀 문제가 아니므로 MSE를 사용하지 않는다. 모델은 연속적인 수익률 값을 예측하는 것이 아니라 Bear, Neutral, Bull 중 하나를 분류한다. 따라서 손실함수는 Cross Entropy Loss이다.

정답 pseudo-label을 $y_i$, 모델이 정답 클래스에 부여한 확률을 $p_{i,y_i}$라고 하면, 기본 Cross Entropy는 다음과 같다.

$$
\mathcal{L}_i(\theta)
= -\log p_{i,y_i}
$$

이는 정답 클래스의 예측 확률이 높을수록 작아지고, 낮을수록 커진다.

클래스 불균형을 보정하기 위해 클래스별 가중치 $w_k$를 적용한다.

$$
\mathcal{L}(\theta)
= \frac{1}{N}\sum_{i=1}^{N}
w_{y_i} \left(-\log p_{i,y_i}\right)
$$

클래스 가중치는 학습 데이터에서 각 클래스가 등장한 횟수 $n_k$에 반비례하도록 설정된다.

$$
w_k = \frac{N}{K n_k}
$$

여기서 $N$은 전체 학습 샘플 수, $K=3$은 클래스 수이다.

또한 Neutral 클래스는 경계가 모호하고 recall이 낮게 나타나는 문제가 있었기 때문에 추가 가중치인 `neutral_boost`를 적용하였다.

$$
w_{\text{Neutral}}
\leftarrow
w_{\text{Neutral}} \times \beta
$$

최종 모델에서는

$$
\beta = 1.2
$$

를 사용하였다.

따라서 최종 손실함수는 다음과 같이 쓸 수 있다.

$$
\mathcal{L}(\theta)
=
-\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\log
\left(
\frac{\exp(z_{i,y_i})}{\sum_{j=1}^{3}\exp(z_{i,j})}
\right)
$$

---

## 5. 최적화 방법: AdamW

모델 파라미터 $\theta$는 손실함수 $\mathcal{L}(\theta)$를 최소화하도록 학습된다.

$$
\theta^*
=
\arg\min_{\theta}
\mathcal{L}(\theta)
$$

본 프로젝트에서는 AdamW optimizer를 사용하였다. AdamW는 Adam의 adaptive learning rate 구조에 decoupled weight decay를 결합한 최적화 방법이다.

일반적인 경사하강법 관점에서 보면 업데이트는 다음과 같은 방향을 가진다.

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

여기서

- $\alpha$: learning rate
- $\lambda$: weight decay coefficient
- $\nabla_{\theta}\mathcal{L}$: 손실함수의 gradient

최종 모델의 주요 설정은 다음과 같다.

$$
\alpha = 10^{-4}
$$

$$
\lambda = 10^{-2}
$$

또한 gradient explosion을 방지하기 위해 gradient clipping을 적용하였다.

$$
\|\nabla_{\theta}\mathcal{L}\| \leq 1.0
$$

---

## 6. 왜 분류 성능만으로는 충분하지 않은가

HMM label은 실제 시장의 절대적 정답이 아니다. HMM state는 통계적 기준에 따라 구성된 pseudo-label이므로, 다음과 같은 문제가 존재한다.

$$
\text{High Classification Accuracy}
\nRightarrow
\text{High Economic Profit}
$$

예를 들어 모델이 HMM state를 잘 맞히더라도, 그 state가 실제 투자 수익률과 약하게 연결되어 있다면 포트폴리오 성과는 개선되지 않을 수 있다.

따라서 본 프로젝트의 핵심 평가는 다음 질문으로 이어져야 한다.

$$
\text{예측된 HMM state probability가 포트폴리오 성과 개선으로 이어지는가?}
$$

즉, 분류 모델은 최종 목적이 아니라 포트폴리오 의사결정을 위한 정보 생성 모듈이다.

---

## 7. 포트폴리오 리밸런싱 문제

포트폴리오에 $m$개의 자산이 있다고 하자. 각 시점 $t$에서의 포트폴리오 비중은 다음과 같다.

$$
w_t =
\begin{bmatrix}
w_{t,1} \\
w_{t,2} \\
\vdots \\
w_{t,m}
\end{bmatrix}
$$

일반적으로 long-only 포트폴리오에서는 다음 제약을 둔다.

$$
\sum_{j=1}^{m} w_{t,j} = 1
$$

$$
w_{t,j} \geq 0
$$

시점 $t+1$의 자산 수익률 벡터를 $r_{t+1}$라고 하면, 포트폴리오 수익률은 다음과 같다.

$$
r_{p,t+1}
=
w_t^\top r_{t+1}
$$

백테스트는 이 과정을 시간 순서대로 반복하여 누적 수익률, 변동성, Sharpe ratio, MDD 등을 계산하는 절차이다.

---

## 8. MVO 기반 자산배분

MVO(Mean-Variance Optimization)는 기대수익률과 위험의 trade-off를 고려하여 자산 비중을 결정하는 방법이다.

각 자산의 기대수익률 벡터를 $\mu$, 공분산 행렬을 $\Sigma$라고 하자.

$$
\mu =
\begin{bmatrix}
\mu_1 \\
\mu_2 \\
\vdots \\
\mu_m
\end{bmatrix}
$$

$$
\Sigma =
\begin{bmatrix}
\sigma_{11} & \sigma_{12} & \cdots & \sigma_{1m} \\
\sigma_{21} & \sigma_{22} & \cdots & \sigma_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
\sigma_{m1} & \sigma_{m2} & \cdots & \sigma_{mm}
\end{bmatrix}
$$

포트폴리오 기대수익률은 다음과 같다.

$$
\mu_p = w^\top \mu
$$

포트폴리오 분산은 다음과 같다.

$$
\sigma_p^2 = w^\top \Sigma w
$$

MVO는 보통 다음 목적함수를 최대화한다.

$$
\max_w
\left(
w^\top \mu
- \lambda w^\top \Sigma w
\right)
$$

또는 같은 의미로 다음 문제를 푼다.

$$
\min_w
\left(
\lambda w^\top \Sigma w
- w^\top \mu
\right)
$$

여기서 $\lambda$는 위험 회피 정도를 나타낸다. $\lambda$가 클수록 위험을 더 많이 줄이는 포트폴리오가 선택된다.

단, 실제 미래 기대수익률 $\mu$는 알 수 없다. 따라서 $\mu$와 $\Sigma$는 과거 데이터로부터 추정한다.

$$
\hat{\mu}
=
\frac{1}{T}
\sum_{t=1}^{T} r_t
$$

$$
\hat{\Sigma}
=
\frac{1}{T-1}
\sum_{t=1}^{T}
(r_t-\hat{\mu})(r_t-\hat{\mu})^\top
$$

즉, MVO에서 사용하는 기대수익률은 알려진 값이 아니라 추정값이다.

---

## 9. 국면 확률과 MVO 비중의 결합

본 프로젝트의 핵심 아이디어는 딥러닝 모델이 출력한 국면 확률을 포트폴리오 비중에 연결하는 것이다.

먼저 각 국면별로 적절한 포트폴리오 비중을 정의하거나 MVO로 계산한다.

$$
w_{\text{Bear}},
\quad
w_{\text{Neutral}},
\quad
w_{\text{Bull}}
$$

모델이 시점 $t$에서 예측한 국면 확률을 다음과 같이 두자.

$$
p_t =
\begin{bmatrix}
p_{t,\text{Bear}} \\
p_{t,\text{Neutral}} \\
p_{t,\text{Bull}}
\end{bmatrix}
$$

그러면 최종 포트폴리오 비중은 hard classification이 아니라 확률 가중 평균으로 구성된다.

$$
w_t
=
p_{t,\text{Bear}} w_{\text{Bear}}
+
p_{t,\text{Neutral}} w_{\text{Neutral}}
+
p_{t,\text{Bull}} w_{\text{Bull}}
$$

이 방식은 "오늘은 Bull이므로 Bull 포트폴리오 100%"처럼 단정적으로 투자하지 않는다. 대신 모델의 불확실성을 반영하여 각 국면별 포트폴리오를 부드럽게 혼합한다.

따라서 딥러닝 모델의 출력은 직접적인 매수/매도 신호가 아니라 포트폴리오 가중치를 조정하는 확률적 정보로 사용된다.

---

## 10. 백테스트를 통한 경제적 검증

시점 $t$에서 결정한 포트폴리오 비중 $w_t$와 다음 기간의 실제 수익률 $r_{t+1}$을 이용하면 포트폴리오 수익률은 다음과 같다.

$$
r_{p,t+1} = w_t^\top r_{t+1}
$$

누적 수익률은 다음과 같이 계산한다.

$$
V_T
=
V_0
\prod_{t=1}^{T}
(1+r_{p,t})
$$

또는 누적 수익률 기준으로는

$$
R_T
=
\prod_{t=1}^{T}
(1+r_{p,t}) - 1
$$

Sharpe ratio는 다음과 같이 정의된다.

$$
\text{Sharpe}
=
\frac{\mathbb{E}[r_p - r_f]}
{\sigma(r_p)}
$$

무위험수익률 $r_f$를 0으로 두면 단순히 다음과 같이 쓸 수 있다.

$$
\text{Sharpe}
\approx
\frac{\mathbb{E}[r_p]}
{\sigma(r_p)}
$$

MDD(Maximum Drawdown)는 누적 가치 $V_t$가 이전 고점 대비 얼마나 하락했는지를 나타낸다.

$$
\text{MDD}
=
\max_t
\left(
\frac{\max_{\tau \leq t} V_{\tau} - V_t}
{\max_{\tau \leq t} V_{\tau}}
\right)
$$

최종적으로 동적 전략은 다음과 같은 정적 benchmark와 비교되어야 한다.

- Equal Weight Portfolio
- 60/40 Portfolio
- Buy-and-Hold
- 정적 MVO Portfolio

이 비교를 통해 예측된 국면 정보가 실제 경제적 성과 개선으로 이어졌는지 확인할 수 있다.

---

## 11. 정적 포트폴리오의 의미와 본 프로젝트의 백테스트 결과

정적 포트폴리오(static portfolio)란 시장 국면, 모델 예측, 변동성 변화 등에 따라 자산 비중을 동적으로 바꾸지 않고, 사전에 정한 비중을 계속 유지하는 포트폴리오를 의미한다.

예를 들어 Buy-and-Hold는 SPY 100% 비중을 유지하는 전략이고, Equal Weight Portfolio는 SPY, QQQ, GLD, TLT를 각각 25%씩 유지하는 전략이다. 60/40 포트폴리오는 주식 60%, 현금 또는 채권 40%와 같은 고정 비중을 유지한다. Regime-Agnostic MVO 역시 훈련 데이터 전체로 한 번 계산한 MVO 비중을 테스트 기간 동안 그대로 유지한다는 점에서 정적 benchmark로 볼 수 있다.

반면 본 프로젝트의 Regime-MVO는 다음과 같이 매 시점의 예측 국면 확률에 따라 비중이 달라진다.

$$
w_t
=
p_{t,\text{Bear}} w_{\text{Bear}}
+
p_{t,\text{Neutral}} w_{\text{Neutral}}
+
p_{t,\text{Bull}} w_{\text{Bull}}
$$

따라서 Regime-MVO는 정적 포트폴리오가 아니라 동적 포트폴리오이다.

실제 백테스트 결과, Regime-MVO가 모든 정적 benchmark보다 높은 누적수익률을 달성한 것은 아니다. 주요 결과는 다음과 같다.

| 전략 | 누적수익률 | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| Equal Weight 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Regime-Agnostic MVO | 64.8% | 1.11 | -20.8% | 1.30 |
| Regime-MVO | 35.3% | 1.10 | -7.2% | 2.16 |

즉, 누적수익률만 보면 Regime-MVO는 Buy-and-Hold, Equal Weight, Regime-Agnostic MVO보다 낮다. 따라서 본 프로젝트의 결과를 "HMM 국면 정보를 사용했기 때문에 정적 포트폴리오보다 수익률이 높아졌다"고 해석해서는 안 된다.

하지만 MDD(Maximum Drawdown)를 기준으로 보면 Regime-MVO는 -7.2%로 가장 낮은 하락폭을 보였다. Buy-and-Hold의 MDD는 -17.0%, 60/40은 -10.4%, Regime-Agnostic MVO는 -20.8%였다. 이는 국면 정보를 반영하지 않은 정적 MVO가 높은 누적수익률을 얻는 대신 큰 하방 위험을 감수한 반면, Regime-MVO는 시장 국면 확률을 이용해 위험 노출을 조절함으로써 손실 폭을 줄였음을 의미한다.

따라서 본 프로젝트의 경제적 의의는 수익률 극대화보다는 하방 위험 관리에 있다.

$$
\text{Regime Information}
\Rightarrow
\text{Dynamic Risk Control}
\Rightarrow
\text{Lower Drawdown}
$$

정리하면, 본 전략은 수익률 1등 전략은 아니지만, HMM 기반 국면 예측을 포트폴리오 리밸런싱에 연결했을 때 정적 benchmark 대비 하방 위험을 줄일 수 있음을 보여준다. 이는 HMM pseudo-label 분류가 단순한 통계적 예측에 머무르지 않고, 위험 관리 목적의 투자 의사결정에 활용될 수 있음을 보여주는 결과이다.

---

## 12. 왜 End-to-End 수익률 최적화를 하지 않았는가

이론적으로는 딥러닝 모델이 직접 포트폴리오 비중을 출력하고, 백테스트 수익률 또는 Sharpe ratio를 직접 최적화하도록 만들 수도 있다.

예를 들어 모델이 직접 $w_t$를 출력한다고 하면,

$$
w_t = h_{\theta}(X_t)
$$

포트폴리오 수익률은

$$
r_{p,t+1}
=
h_{\theta}(X_t)^\top r_{t+1}
$$

이고, 누적 수익률을 최대화하는 objective는 다음과 같이 쓸 수 있다.

$$
\max_{\theta}
\sum_{t=1}^{T}
h_{\theta}(X_t)^\top r_{t+1}
$$

또는 Sharpe ratio를 직접 최대화할 수도 있다.

$$
\max_{\theta}
\frac{\mathbb{E}[r_p]}
{\sigma(r_p)}
$$

그러나 본 프로젝트에서는 이 방식을 채택하지 않았다. 이유는 다음과 같다.

첫째, 금융 수익률은 noise가 매우 크고 sample 수가 제한적이다. End-to-end 방식은 train 구간의 우연한 수익 패턴에 과최적화될 위험이 크다.

둘째, Sharpe ratio나 MDD 같은 투자 성과 지표는 cross entropy에 비해 최적화가 불안정하다. 특히 Sharpe ratio는 변동성이 작게 추정될 때 gradient가 불안정해질 수 있다.

셋째, 본 프로젝트의 목적은 black-box trading model을 만드는 것이 아니라 HMM으로 정의한 시장 국면 정보가 포트폴리오 리밸런싱에 유용한지를 검증하는 것이다. 따라서 국면 예측 모듈과 자산배분 모듈을 분리하는 것이 해석 가능성 측면에서 더 적절하다.

즉, end-to-end를 사용하지 않은 것은 수익률 최적화를 고려하지 않았기 때문이 아니라, 제한된 데이터와 높은 금융 noise 환경에서 과최적화 위험을 줄이고 연구 구조를 명확히 하기 위한 선택이다.

---

## 13. 모델 구조 선택: Transformer를 최종 구조로 채택하지 않은 이유

본 프로젝트에서는 Transformer 계열 모델도 검토되었다. 특히 0528 수정 결과의 `모델 구조+한계` 문서에는 SPY 데이터 기반 Transformer 구조가 정리되어 있다. 해당 구조는 10차원 입력 피처를 Transformer가 처리하기 쉬운 $d_{\text{model}}=64$ 차원으로 투영하고, 시간 순서를 반영하기 위한 positional embedding을 더한 뒤, multi-head self-attention과 feed-forward layer를 통과시킨다. 이후 30거래일 전체 출력의 평균을 사용해 시퀀스를 하나의 벡터로 요약하고, 마지막 classifier를 통해 Bear, Neutral, Bull logits을 출력한다.

즉, Transformer를 고려하지 않은 것이 아니라, 실험 및 구조 검토 후 최종 모델로 Conv1D+LSTM을 채택한 것이다. 이 절의 목적은 왜 Transformer가 아닌 Conv1D+LSTM이 최종 파이프라인에 더 적합하다고 판단했는지를 설명하는 데 있다.

Transformer는 시계열의 모든 시점 간 관계를 self-attention으로 직접 학습할 수 있다는 장점이 있다. 일반적으로 입력 시퀀스를

$$
X =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_T
\end{bmatrix}
$$

라고 하면, Transformer의 self-attention은 각 시점 간 유사도를 계산한다.

$$
\text{Attention}(Q,K,V)
=
\text{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

이 구조는 긴 문장이나 대규모 시계열처럼 충분한 데이터가 있고 장거리 의존성이 중요한 문제에서 강력하다. 하지만 본 프로젝트의 금융 시계열에서는 다음과 같은 이유로 Conv1D+LSTM이 더 적합하다고 판단하였다.

첫째, 금융 데이터는 signal-to-noise ratio가 낮다. 관측 수익률 또는 피처를 다음과 같이 생각할 수 있다.

$$
x_t = s_t + \epsilon_t
$$

여기서 $s_t$는 시장 국면과 관련된 잠재 신호이고, $\epsilon_t$는 단기 noise이다. 금융 데이터에서는 $\epsilon_t$의 비중이 매우 크기 때문에, 모델이 너무 유연하면 실제 신호 $s_t$보다 우연한 noise 패턴을 학습할 위험이 커진다.

둘째, Transformer는 모든 시점 간 pairwise relation을 학습하기 때문에 표현력이 크다. 시퀀스 길이가 $T$일 때 self-attention은 대략 $T^2$개의 시점 간 관계를 고려한다. 본 프로젝트의 입력 길이는 30일로 길지 않고, 학습 샘플 수 역시 제한적이다. 이런 상황에서는 Transformer의 높은 표현력이 장점이 아니라 과적합 위험으로 작용할 수 있다.

일반화 관점에서도 모델 capacity가 커질수록 제한된 표본에서 추정 오차가 커질 수 있다. 매우 단순화하면 일반화 오차는 다음과 같은 방향성을 가진다.

$$
\text{Generalization Gap}
\propto
O\left(\sqrt{\frac{\mathcal{C}}{N}}\right)
$$

여기서 $\mathcal{C}$는 모델 복잡도, $N$은 학습 샘플 수이다. $N$이 충분히 크지 않은 상황에서 $\mathcal{C}$만 커지면 train 성능은 좋아져도 validation/test 성능은 악화될 수 있다.

셋째, LSTM은 금융 시계열에 유리한 inductive bias를 가진다. LSTM은 다음과 같이 이전 hidden state를 누적하면서 정보를 갱신한다.

$$
h_t = F_{\theta}(x_t, h_{t-1})
$$

그리고 gate 구조를 통해 어떤 정보를 유지하고 어떤 정보를 잊을지 학습한다.

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

이 구조는 모든 시점을 동등하게 보는 것이 아니라, 시간 순서에 따라 정보를 누적하고 불필요한 단기 변동을 일부 걸러내는 역할을 한다. 이런 의미에서 LSTM은 금융 데이터의 random fluctuation에 대해 일종의 temporal smoothing 또는 low-pass filter와 유사한 inductive bias를 가진다고 해석할 수 있다. 이는 완전한 수학적 저역통과 필터라는 뜻은 아니지만, 짧은 기간의 noise보다 30일 동안 누적되는 국면적 흐름을 더 안정적으로 반영할 수 있다는 의미이다.

넷째, 본 프로젝트의 실험에서도 더 복잡한 attention 계열 구조가 항상 개선을 만들지는 않았다. BiLSTM + Attention 실험은 이론적으로 더 강한 구조였지만, test accuracy가 49.5%로 baseline보다 낮았다. 또한 LSTM 전체 시점 평균(Global Average Pooling)을 사용한 실험도 최근 신호를 희석하여 성능이 저하되었다. 이는 30일 금융 시계열에서 "모든 시점을 넓게 보는 구조"가 항상 유리한 것은 아니며, 최근 정보와 순차적 누적을 반영하는 LSTM 구조가 더 안정적일 수 있음을 보여준다.

따라서 본 프로젝트에서 Transformer를 최종 구조로 채택하지 않은 이유는 Transformer가 일반적으로 열등하기 때문이 아니다. 오히려 본 데이터의 특성, 즉 작은 표본 수, 높은 noise, 짧은 입력 길이, pseudo-label 기반 학습이라는 조건에서는 Conv1D+LSTM이 더 보수적이고 안정적인 선택이었다.

정리하면 다음과 같다.

$$
\text{High-noise Financial Data}
+
\text{Small Sample Size}
\Rightarrow
\text{Prefer Stronger Temporal Inductive Bias}
$$

$$
\text{Conv1D+LSTM}
\Rightarrow
\text{Local Pattern Extraction}
+
\text{Sequential Smoothing}
+
\text{Lower Overfitting Risk}
$$

발표에서는 다음과 같이 설명할 수 있다.

> Transformer 계열 구조도 검토했지만, 금융 데이터는 랜덤성이 크고 본 프로젝트의 학습 표본이 제한적이기 때문에 모든 시점 간 관계를 자유롭게 학습하는 self-attention 구조는 과거 구간의 noise를 과적합할 위험이 있다. 반면 Conv1D+LSTM은 국소 패턴을 먼저 추출하고 시간 순서대로 정보를 누적하므로, 짧은 금융 시계열에서 더 안정적인 low-pass 성격의 inductive bias를 제공한다. 따라서 본 프로젝트에서는 Transformer를 최종 구조로 채택하기보다 Conv1D+LSTM 기반 국면 분류기를 최종 모델로 사용하였다.

---

## 14. 전체 파이프라인의 수학적 요약

전체 프로젝트는 다음과 같이 정리할 수 있다.

### Step 1. HMM pseudo-label 생성

$$
\hat{y}_{t}^{HMM}
=
g_{\phi}(x_{1:t})
$$

### Step 2. 딥러닝 기반 국면 확률 예측

$$
p_t
=
f_{\theta}(X_t)
=
\begin{bmatrix}
P(\text{Bear} \mid X_t) \\
P(\text{Neutral} \mid X_t) \\
P(\text{Bull} \mid X_t)
\end{bmatrix}
$$

### Step 3. Weighted Cross Entropy 최소화

$$
\theta^*
=
\arg\min_{\theta}
\left[
-\frac{1}{N}
\sum_{i=1}^{N}
w_{y_i}
\log p_{i,y_i}
\right]
$$

### Step 4. 국면별 포트폴리오 비중 계산

$$
w_{\text{regime}}^*
=
\arg\max_w
\left(
w^\top \hat{\mu}_{\text{regime}}
- \lambda w^\top \hat{\Sigma}_{\text{regime}} w
\right)
$$

### Step 5. 예측 확률 기반 동적 리밸런싱

$$
w_t
=
p_{t,\text{Bear}} w_{\text{Bear}}
+
p_{t,\text{Neutral}} w_{\text{Neutral}}
+
p_{t,\text{Bull}} w_{\text{Bull}}
$$

### Step 6. 실제 수익률 기반 백테스트

$$
r_{p,t+1}
=
w_t^\top r_{t+1}
$$

$$
R_T
=
\prod_{t=1}^{T}
(1+r_{p,t}) - 1
$$

---

## 15. 결론

본 프로젝트에서 딥러닝 모델의 직접적인 학습 task는 HMM 기반 시장 국면 pseudo-label을 예측하는 3-class classification이다. 손실함수는 weighted cross entropy이며, AdamW optimizer를 통해 모델 파라미터를 최적화하였다.

그러나 HMM state는 절대적인 시장 정답이 아니라 통계적 pseudo-label이므로, 해당 state를 잘 예측했다는 사실만으로 경제적 유효성을 주장할 수 없다. 따라서 본 프로젝트의 최종 목적은 예측된 국면 확률을 포트폴리오 리밸런싱에 연결하고, 정적 포트폴리오 대비 백테스트 성과가 어떻게 달라지는지 확인하는 것이다.

백테스트 결과, Regime-MVO는 정적 benchmark 대비 누적수익률을 항상 개선하지는 못했다. 그러나 MDD를 -7.2%로 낮추어 Buy-and-Hold, 60/40, Regime-Agnostic MVO보다 우수한 하방 위험 관리 성능을 보였다. 따라서 본 프로젝트의 핵심 의의는 수익률 극대화가 아니라, HMM 기반 국면 정보를 이용해 포트폴리오의 위험 노출을 조절하고 drawdown을 완화했다는 점에 있다.

즉, 본 연구는 다음과 같은 구조로 이해하는 것이 가장 적절하다.

$$
\text{HMM pseudo-label}
\rightarrow
\text{Deep Learning Regime Prediction}
\rightarrow
\text{Probability-based Portfolio Rebalancing}
\rightarrow
\text{Backtest against Static Benchmarks}
$$

따라서 분류와 백테스트는 별개의 독립적 task라기보다, 시장 국면 정보가 실제 투자 성과로 이어지는지를 검증하기 위한 하나의 연결된 연구 파이프라인이다.
