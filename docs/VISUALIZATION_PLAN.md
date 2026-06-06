# Binary Soft MVO 실험 이후 시각화 계획

## 1. 왜 그림을 업데이트했나

기존 final figure는 오래된 3-class Regime-MVO 결과를 중심으로 만들어져 있었다.

```text
3-class Regime-MVO original
cum_ret = 35.3%
Sharpe  = 1.10
MDD     = -7.2%
Calmar  = 2.16
```

하지만 troubleshooting 이후 현재 가장 강한 전략은 다음으로 바뀌었다.

```text
Binary Regime-MVO Soft, cap 40%
cum_ret = 53.7%
Sharpe  = 1.48
MDD     = -9.0%
Calmar  = 2.55
```

따라서 "Regime-MVO는 수익률 1등 전략이 아니다"라는 예전 메시지는 더 이상 최종 결론으로 쓰면 안 된다. 새 결론은 다음에 가깝다.

> Binary soft-label 국면 예측과 capped 2-Regime MVO를 결합하면 EW 1/N 및 3-class capped MVO보다 누적수익률과 Calmar가 개선되고, MDD는 비슷한 수준으로 유지된다.

## 2. 현재 Figure 상태

| Figure | 상태 | 처리 |
|---|---|---|
| `fig01_pipeline.png` | 대체로 유효 | 필요하면 Binary Soft Label -> 2-Regime MVO 흐름을 더 명시적으로 업데이트 |
| `fig02_related_work.png` | 유효 | 수정 필요 없음 |
| `fig03_static_dynamic_backtest.png` | 업데이트 완료 | Binary Regime-MVO Soft cap 40%를 메인 동적 전략으로 사용 |
| `fig03_main_result.png` | 업데이트 완료 | 새 최종 전략의 핵심 성과 요약 |
| `fig04_classification_performance.png` | 업데이트 완료 | Binary Hard, Binary Soft, LR, RF 비교 |
| `fig05_confusion_matrix.png` | 업데이트 완료 | Binary Soft Label confusion matrix |
| `fig07_ablation.png` | 업데이트 완료 | Buy & Hold, EW 1/N, 3-class capped MVO, Binary Soft MVO 비교 |
| `fig09_binary_mvo_weights.png` | 추가 완료 | MVO cap이 extreme weight를 줄이는 효과 시각화 |

## 3. 최종 발표에 필요한 그림

### Figure A: 최종 전략 성과 요약

목적:

최종 전략을 주요 benchmark와 직접 비교한다.

| Strategy | CumRet | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

추천 형태:

- 누적수익률, MDD, Calmar를 나눠 보여주는 3-panel bar chart
- Binary Regime-MVO Soft cap 40% 강조
- EW 1/N은 가장 강한 단순 benchmark이므로 반드시 같이 표시

### Figure B: 누적수익률과 Drawdown 경로

목적:

최종 전략이 단순히 표에서만 좋아 보이는 것이 아니라, 실제 테스트 기간 경로에서도 어떤 움직임을 보이는지 보여준다.

비교 전략:

- Buy & Hold
- EW 1/N
- 3-class Regime-MVO cap 40%
- Binary Regime-MVO Soft cap 40%

추천 형태:

- 위쪽: 누적수익률 경로
- 아래쪽: drawdown 경로
- 테스트 기간: 2024-04-15 ~ 2026-05-15

### Figure C: 분류 성능 개선

목적:

왜 3-class에서 binary soft label로 넘어갔는지 설명한다.

| Model | Balanced Accuracy | Bear Recall |
|---|---:|---:|
| 3-class hard label | 51.9% | 60.5% |
| Binary hard label | 70.2% | 58.1% |
| Binary soft label | 72.4% | 67.4% |
| LR baseline | 61.4% | 32.6% |
| RF baseline | 66.3% | 53.5% |

추천 형태:

- 가로 bar chart 또는 compact table
- 하방 위험 탐지가 목적이므로 Bear Recall 강조

### Figure D: Binary MVO 비중 구성

목적:

MVO cap이 왜 필요한지 시각적으로 보여준다.

| Cap | Non-Bear MVO | Bear MVO |
|---:|---|---|
| 100% | SPY 100.0% | TLT 100.0% |
| 50% | SPY 50.0%, QQQ 49.8%, GLD 0.2% | QQQ 7.3%, GLD 42.7%, TLT 50.0% |
| 40% | SPY 40.0%, QQQ 40.0%, GLD 17.3%, TLT 2.7% | QQQ 20.0%, GLD 40.0%, TLT 40.0% |

추천 형태:

- Stacked bar chart
- Non-Bear / Bear 패널 분리
- 특정 자산 몰빵을 줄이는 효과를 설명하는 데 사용

## 4. 현재 Figure 생성 스크립트

정리된 final figure set은 다음 명령으로 생성한다.

```bash
python3 scripts/visualize_binary_mvo_results.py
```

이 스크립트가 다시 생성하는 파일:

- `outputs/figures/final/fig03_static_dynamic_backtest.png`
- `outputs/figures/final/fig03_main_result.png`
- `outputs/figures/final/fig04_classification_performance.png`
- `outputs/figures/final/fig05_confusion_matrix.png`
- `outputs/figures/final/fig07_ablation.png`
- `outputs/figures/final/fig09_binary_mvo_weights.png`

나머지 파이프라인/관련연구 그림은 아래 스크립트로 생성한다.

```bash
python3 scripts/visualize_pipeline.py
python3 scripts/visualize_related_work.py
```

## 5. 현재 문서 동기화 상태

다음 문서들은 현재 최종 전략 기준으로 동기화되어 있다.

- `README.md`
- `docs/PRESENTATION_OUTLINE.md`
- `docs/TROUBLESHOOTING.md`
- `docs/CONCEPTUAL_MATH_REPORT_NOTION.md`
- `outputs/figures/README.md`

현재 최종 전략:

```text
Binary Soft Label + 2-Regime MVO + cap 40%
```
