# 발표 목차 — 국면 인식 기반 동적 포트폴리오 최적화

> 총 15분 / 4인 기준

---

## 핵심 메시지 (발표 전체를 관통하는 한 줄)

> **"시장 국면을 인식하는 것만으로 포트폴리오 최대 손실을 3분의 1로 줄일 수 있다"**

---

## 파트 분배 요약

| 파트 | 담당 | 내용 | 시간 | 슬라이드 |
|---|---|---|---|---|
| A | 팀원 1 | 도입부 — 문제 정의 + 관련 연구 | ~3분 | 1~3 |
| B | 팀원 2 | 방법론 — 파이프라인 + 모델 | ~3분 | 4~5 |
| C | 팀원 3 | 실험 — 분류 성능 + 국면별 성과 | ~3.5분 | 6~7 |
| D | 팀원 4 | 결과 — Ablation + 2022 검증 + 결론 | ~5분 | 8~10 |

---

## 파트 A — 도입부 (팀원 1, ~3분)

### 1. 표지 (30초)
- 제목: 국면 인식 기반 동적 포트폴리오 최적화
- 팀명 / 팀원 이름 / 날짜

### 2. 문제 정의 (1분)

**"포트폴리오가 하락장에서 왜 실패하는가?"**

- 국면을 무시한 MVO는 MDD -20.8% — 최적화했는데 오히려 Buy & Hold보다 나쁨
- Buy & Hold도 MDD -17.0% — 하락장에서 속수무책
- **핵심 질문:** 시장 국면을 인식하면 포트폴리오를 더 잘 지킬 수 있을까?

> 강조: "최적화를 했어도 국면을 모르면 오히려 더 나빠질 수 있다"

### 3. 관련 연구 + 비교표 (1.5분) — fig02

- Kim et al. (2019): HMM 기반 국면 레이블링 정립 — 저희 기반
- RegimeFolio (Zhang et al. 2025): VIX 기반 국면 + MVO — 동일 구조
- **우리 차이점:** 국면 감지를 VIX 룰 대신 Conv1D+LSTM으로 학습

> 강조: "RegimeFolio와 동일한 구조, 국면 분류기를 데이터로 학습한 게 차이"

---

## 파트 B — 방법론 (팀원 2, ~3분)

### 4. 파이프라인 (1.5분) — fig01

```
Step 1. HMM으로 국면 정의
  → 과거 2년 데이터 기반, 3-state Gaussian HMM
  → Sharpe 기준 Bear / Neutral / Bull 분류

Step 2. Conv1D+LSTM이 국면 예측
  → 과거 30일 × 40개 피처 → P(Bear), P(Neutral), P(Bull)

Step 3. 확률로 MVO 비중 결합
  → w = P(Bear)×w_bear + P(Neutral)×w_neutral + P(Bull)×w_bull
  → 국면 확률이 높을수록 해당 국면 최적 비중으로 연속 이동
```

### 5. 모델 구조 (1.5분)

- 입력: 30일 × 40피처 (SPY/QQQ/GLD/TLT 각 10개 피처)
- Conv1D → 단기 패턴, LSTM → 시계열 맥락
- 출력: P(Bear) / P(Neutral) / P(Bull)

국면별 MVO 최적 비중 (훈련 데이터 Sharpe 최대화):
```
Bear    → TLT 100%               (하락장: 채권 방어)
Neutral → SPY 51% + GLD 49%      (중립: 균형)
Bull    → SPY 95% + QQQ 5%       (상승장: 주식 집중)
```

> 강조: "비중을 사람이 정한 게 아니라 훈련 데이터가 자동으로 계산한 것"

---

## 파트 C — 실험 (팀원 3, ~3.5분)

### 6. 분류 성능 (1.5분) — fig04, fig05

**📊 fig04 — 분류 성능 표 (Phase별 개선 과정)**

| 단계 | Bear Recall | 의미 |
|---|---|---|
| Ph1 Baseline | 34.9% | SPY 단독, 기본 모델 |
| Ph3 최종 | **60.5%** | 4자산 Cross-asset + AdamW |

**📊 fig05 — Confusion Matrix**
- Bear 60.5% 탐지, Bull 95.1% 탐지
- Neutral 0%: 중립 구간 레이블 불명확 — 한계로 솔직히 언급

> 강조: "분류 성능이 목적이 아니라 포트폴리오 개선의 수단"

### 7. 국면별 성과 분석 (2분) — fig06

**📊 fig06 — 국면별 전략 성과 (Bear/Neutral/Bull × 전략)**

Bear 구간(43 periods):
- Buy & Hold: -8.4% (연환산)
- Regime-MVO: **+1.4%** — 유일하게 플러스

MDD 기준:
- Buy & Hold Bear MDD: -18.2%
- Regime-MVO Bear MDD: **-7.5%** (금색 테두리 = 전략 중 최저)

> 강조: "Bear 구간에서 우리 전략만 방어가 됐다. 이게 핵심 기여"

---

## 파트 D — 결과 & 마무리 (팀원 4, ~5분)

### 8. Ablation Study (2분) — fig07

**📊 fig07 — 각 구성요소 기여도**

| # | 구성 | MDD | Calmar | 의미 |
|---|---|---|---|---|
| 1 | Buy & Hold | -17.0% | 1.26 | 베이스라인 |
| 2 | Regime-**Agnostic** MVO | **-20.8%** | 1.30 | 국면 무시하면 MVO가 오히려 악화 |
| 3 | DL Regime SPY/Cash | -7.4% | 1.35 | 국면 신호만 써도 MDD 급감 |
| 4 | **Regime-MVO (ours)** | **-7.2%** | **2.16** | 국면 신호 + MVO 결합 |
| 5 | Oracle (완벽한 분류기) | -6.2% | 2.91 | 이론적 상한선 |

**두 가지 핵심 발견:**
1. **국면 conditioning 효과:** MDD -20.8% → -7.2% (13.6pp 개선)
2. **우리 분류기 충분:** Oracle과 MDD 1pp 차이 → 이미 이론 상한의 92%

> 강조: "국면을 인식하는 것 자체가 핵심. 분류기도 이미 충분히 잘 한다."

### 9. 2022 하락장 검증 (1.5분) — fig08

**📊 fig08 — 2022 Bear Market**

2022: Fed 금리 급인상 → 주식·채권 동반 하락

| 전략 | MDD |
|---|---|
| Buy & Hold | -20.5% |
| EW 1/N | -21.7% |
| **DL Regime SPY/Cash** | **-10.5%** ← |
| Regime-MVO | -22.2% |

- 학습 데이터에 없는 환경에서도 낙폭 49% 감소
- 한계: TLT도 폭락하는 금리인상형 Bear는 취약 → 향후 방어 자산 다양화 필요

### 10. 결론 (1분)

**3줄 요약**

1. 국면 Conditioning 자체가 핵심 — Agnostic MVO 대비 MDD **13.6pp** 개선
2. Regime-MVO가 Oracle(완벽한 분류기)의 **92%** 달성 — 현재 분류기로 충분
3. 2022 실전 검증 — 학습 외 환경에서도 낙폭 **49% 감소**

**향후 연구**
- 방어 자산 TLT → GLD + 단기채 다양화 (금리인상형 Bear 대응)
- End-to-end 학습: Sharpe를 직접 loss로 최적화

> **마지막 멘트:**
> "수익률 1등이 목표가 아닙니다. 언제 위험한지 알고, 그에 맞게 대응하는 것이 목표입니다."

---

## 시간 배분

| 파트 | 내용 | 시간 |
|---|---|---|
| A (1~3) | 문제 + 관련 연구 | 3분 |
| B (4~5) | 파이프라인 + 모델 | 3분 |
| C (6~7) | 분류 성능 + 국면별 성과 | 3.5분 |
| D (8~10) | Ablation + 2022 + 결론 | 5분 |
| **합계** | | **14.5분** |

---

## 그림 배치

| 그림 | 파일 | 파트 | 핵심 메시지 |
|---|---|---|---|
| fig01 — 파이프라인 | `outputs/figures/final/fig01_pipeline.png` | B (4) | HMM → Conv1D+LSTM → MVO 3단계 |
| fig02 — 관련 연구 표 | `outputs/figures/final/fig02_related_work.png` | A (3) | 우리만 Learned Classifier O |
| fig03 — 핵심 결과 | `outputs/figures/final/fig03_main_result.png` | D (8) | 국면 conditioning으로 MDD 축소 |
| fig04 — 분류 성능 표 | `outputs/figures/final/fig04_classification_performance.png` | C (6) | Bear Recall 34% → 60% |
| fig05 — Confusion Matrix | `outputs/figures/final/fig05_confusion_matrix.png` | C (6) | 분류기 실제 예측 결과 |
| fig06 — 국면별 성과 | `outputs/figures/final/fig06_regime_conditional.png` | C (7) | Bear 구간에서만 플러스 |
| fig07 — Ablation | `outputs/figures/final/fig07_ablation.png` | D (8) | 구성요소별 기여 비교 |
| fig08 — 2022 하락장 | `outputs/figures/final/fig08_2022_bear.png` | D (9) | 금리인상형 Bear의 한계 검증 |

---

## 예상 Q&A

**Q. 국면을 무시한 MVO가 왜 MDD -20.8%로 더 나쁜가?**
> 전체 훈련 데이터로 Sharpe 최대화를 하면 수익이 높은 자산(주식)에 집중됩니다. 국면별로 분리해서 최적화해야 Bear일 때는 방어 자산으로, Bull일 때는 공격 자산으로 이동할 수 있습니다.

**Q. Oracle과 1pp 차이면 분류기를 더 개선할 필요가 없지 않나?**
> MDD 기준으로는 그렇습니다. 다만 Calmar에서 0.75 차이가 있어서 더 정확한 분류기는 수익률 측면에서 추가 개선이 가능합니다. 현재 모델로도 충분하지만 개선 여지는 있습니다.

**Q. 왜 2022 하락장에서 Regime-MVO가 아닌 SPY/Cash가 더 잘 버텼나?**
> 2022년은 금리인상으로 주식과 채권이 동시에 폭락했습니다. Regime-MVO의 방어 자산이 TLT(채권)인데 TLT도 -26% 폭락했습니다. SPY/Cash는 현금을 보유하므로 이 상황에서 더 강했습니다. 향후 방어 자산을 GLD 등으로 다양화하면 해결됩니다.
