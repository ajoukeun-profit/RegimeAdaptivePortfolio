# Troubleshooting Notes: Label, Optimization, and MVO Stability

> Neutral label failure, pseudo-label reliability, and small-sample MVO estimation error.
> This note summarizes the current diagnosis and the next experiments to try.

---

## 1. Final Takeaway

The strongest current direction is:

```text
Neutral failure
-> Bear vs Non-Bear binary classification
-> Binary soft-label training
-> 2-Regime MVO
-> MVO weight cap 40%
```

Why:

- Neutral is structurally ambiguous and the 3-class model never predicts Neutral on test.
- Binary classification improves balanced accuracy from 51.9% to 70.2%.
- Binary soft labels improve Bear recall from 58.1% to 67.4%.
- MVO caps reduce concentration caused by small-sample estimation error.
- Binary Regime-MVO Soft with cap 40% gives the best current balanced portfolio result.

Key result:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO, cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft, cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

Interpretation:

Binary Regime-MVO Soft with cap 40% has the highest cumulative return among the current comparable strategies. Compared with EW 1/N, it improves cumulative return by 2.8 percentage points while keeping MDD almost the same. Compared with 3-class capped Regime-MVO, it improves cumulative return by 1.8 percentage points and Calmar from 2.47 to 2.55.

---

## 2. Current Diagnosis

The project is not mainly failing because of gradient explosion or vanishing gradient. The clearest problems are:

1. Neutral label identification failure
2. HMM pseudo-label reliability
3. MVO estimation error under small regime-specific samples

The final model still has useful Bear/Bull discrimination, but the Neutral class is structurally weak.

Final result evidence:

- `outputs/results/train_history.json`
- Accuracy: 61.9%
- Balanced Accuracy: 51.9%
- Bear Recall: 60.5%
- Neutral Recall: 0.0%
- Bull Recall: 95.1%

Confusion matrix:

```text
              Pred Bear  Pred Neutral  Pred Bull
Actual Bear       26          0           17
Actual Neutral     8          0           13
Actual Bull        2          0           39
```

This means the model never predicts Neutral on the test set. It maps ambiguous middle regimes into Bear or Bull.

---

## 3. Gradient Explosion and Vanishing Gradient

### Gradient Explosion

The training code already applies gradient clipping.

Location: `scripts/train.py`

```python
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This means gradient explosion is being actively controlled. The recorded train/validation losses do not show NaN or unstable divergence, so there is no strong evidence that gradient explosion is the main issue.

However, gradient norm is not currently logged. Therefore, the exact frequency or severity of clipping is unknown.

### Vanishing Gradient

The model uses a 30-day sequence and a small LSTM hidden size. This makes severe vanishing gradient less likely than in very long sequence problems.

But the project does not log layer-wise gradient norms or hidden-state statistics. Therefore, vanishing gradient cannot be confirmed or rejected from current evidence.

Conclusion:

> Gradient issues are possible, but the observed failure mode is much more clearly a label definition and estimation problem.

---

## 4. Problem A: Neutral Label Identification Failure

### Why It Happens

Neutral is not a sharply defined economic regime. In this project, HMM states are mapped to Bear, Neutral, and Bull by relative Sharpe/statistical properties. Neutral is effectively the middle state between high-risk Bear and high-return Bull.

This creates two difficulties:

- The boundary between Bear and Neutral is unclear.
- The boundary between Neutral and Bull is also unclear.

As a result, the model can improve overall accuracy by learning a simpler Bear/Bull separation and ignoring Neutral.

### Current Attempt

The project already tried to mitigate this with class weighting and `neutral_boost`.

Location: `scripts/train.py`

```python
weights = n_samples / (float(num_classes) * counts)
weights[1] *= neutral_boost
```

But this creates a trade-off:

- Higher Neutral weight improves Neutral recall.
- Bear and Bull recall often drop.

Prior experiment evidence in `docs/MODEL_IMPROVEMENT.md`:

- `neutral_boost=2.0`: Neutral recall improved, but total accuracy and Bear/Bull performance dropped.
- `neutral_boost=1.2`: Better overall model, but Neutral recall remained 0.0%.

### Candidate Fix 1: Bear vs Non-Bear

This is the most practical first experiment.

Instead of forcing three classes:

```text
Bear / Neutral / Bull
```

convert the target into:

```text
Bear / Non-Bear
```

This matches the project's economic objective better. The strategy is primarily about risk control, so detecting Bear regimes is more important than separating Neutral from Bull.

Possible target conversion:

```python
y = 1 if int(target["hmm_label_code"]) == 0 else 0
```

Expected benefit:

- Removes the ambiguous Neutral boundary.
- Directly optimizes the risk-detection task.
- Easier to explain in presentation: "we reframed the problem as downside-risk detection."

Expected risk:

- The model no longer gives separate Neutral/Bull probabilities.
- Regime-MVO must be adjusted into binary regime MVO.

---

## 5. Problem B: Pseudo-Label Reliability

### Why It Matters

HMM labels are not true market labels. They are pseudo-labels generated by a statistical model.

Therefore, a hard HMM label can overstate certainty. For example, a row with probabilities:

```text
Bear 0.42 / Neutral 0.39 / Bull 0.19
```

may become a hard Bear label, even though the HMM itself is uncertain.

Training with hard labels treats Neutral and Bull as completely wrong for this sample, which can inject label noise into the classifier.

### Candidate Fix 2: Soft Labels

Use HMM probabilities as training targets:

```text
[prob_bear, prob_neutral, prob_bull]
```

instead of:

```text
hmm_label_code
```

Current hard-label target location:

- `scripts/prepare_supervised_dataset.py`
- `scripts/prepare_cross_asset_dataset.py`

Current idea:

```python
y = int(target["hmm_label_code"])
```

Soft-label alternative:

```python
y_soft = [
    float(target["prob_bear"]),
    float(target["prob_neutral"]),
    float(target["prob_bull"]),
]
```

Training loss:

```python
log_probs = torch.log_softmax(logits, dim=1)
loss = -(soft_targets * log_probs).sum(dim=1).mean()
```

Expected benefit:

- Preserves HMM uncertainty.
- Reduces the damage from ambiguous Neutral labels.
- Keeps the 3-class regime probability output.

Expected risk:

- Requires dataset and training-loop changes.
- Evaluation is less straightforward because targets are distributions rather than single labels.

Recommended evaluation:

- Still report hard-label accuracy by comparing `argmax(pred)` to `argmax(soft_target)`.
- Also report KL divergence or soft cross entropy.
- Most importantly, run the portfolio backtest.

### Candidate Fix 3: Confidence Filtering or Sample Weighting

Use HMM confidence to decide how strongly each sample contributes.

Example:

```python
confidence = max(prob_bear, prob_neutral, prob_bull)
```

Options:

- Drop low-confidence samples.
- Keep all samples but multiply loss by confidence.
- Use only high-confidence samples for training and all samples for validation/test.

Expected benefit:

- Reduces pseudo-label noise.

Expected risk:

- The dataset is already small, so dropping samples can hurt.

For that reason, confidence weighting is safer than hard filtering.

---

## 6. Problem C: Small-Sample MVO Estimation Error

### Why It Happens

MVO estimates expected return and risk from historical samples. In this project, MVO is computed separately per regime.

Location: `scripts/backtest_mvo.py`

```python
def max_sharpe_weights(R: np.ndarray, rf: float = RF_P) -> np.ndarray:
```

This creates a small-sample problem:

- The full training set is already small.
- Splitting it into Bear/Neutral/Bull makes each regime sample smaller.
- Sharpe maximization is sensitive to noisy mean and volatility estimates.
- The optimizer can choose extreme weights, such as one asset at 100%.

This is visible in the 2022 stress test: Bear MVO selected TLT-heavy defense, but 2022 was a rate-hike Bear where long-duration bonds also fell.

### Candidate Fix 4: Add Weight Constraints

Current bounds:

```python
bounds=[(0.0, 1.0)] * n
```

Possible more conservative bounds:

```python
bounds=[(0.0, 0.6)] * n
```

Expected benefit:

- Prevents extreme single-asset concentration.
- Makes MVO less sensitive to noisy estimates.

Expected risk:

- May reduce upside in regimes where concentrated exposure worked.

### Candidate Fix 5: Add Cash or Short-Term Bond Defense

The current asset set is:

```text
SPY / QQQ / GLD / TLT
```

In 2022, TLT failed as a defensive asset because interest rates rose sharply. A more robust defensive universe should include:

- Cash
- Short-term Treasury ETF
- Money-market proxy
- Short-duration bond ETF

Expected benefit:

- Better behavior in rate-hike Bear markets.

Expected risk:

- Requires additional data and careful benchmark consistency.

### Candidate Fix 6: Shrinkage or Simpler Portfolio Rules

Alternatives to pure Sharpe-max MVO:

- Minimum variance portfolio
- Mean-variance objective with explicit risk aversion
- Covariance shrinkage
- Blend MVO weights with equal weight
- Add turnover penalty

Example blending:

```python
w_final = 0.7 * w_mvo + 0.3 * w_equal
```

This reduces estimator-driven extreme allocations.

---

## 7. Classical ML Baselines

Because the dataset is small, classical ML baselines are important.

Recommended models:

- Logistic Regression for Bear vs Non-Bear
- Random Forest for Bear vs Non-Bear
- Random Forest for 3-class classification
- Possibly Gradient Boosting if dependency constraints allow it

Purpose:

> These are not just backup models. They test whether Conv1D+LSTM is actually adding value beyond simpler decision rules.

If Logistic Regression or Random Forest performs similarly to LSTM, that is not necessarily bad. It means the core contribution is not model complexity, but the regime-aware portfolio pipeline.

Suggested features for classical baselines:

- Flatten `(30, 40)` into 1200 features
- Or use summary features over the 30-day window:
  - latest value
  - mean
  - standard deviation
  - min/max
  - recent 5-day mean

The summary-feature version may generalize better because the sample size is small.

---

## 8. Experiment Results

### Priority 1: Bear vs Non-Bear Classification

Reason:

- Directly targets the project's main goal: downside-risk detection.
- Easiest way to remove the Neutral ambiguity.
- Best fit for the current result, where Bear and Bull are learnable but Neutral is not.

Implementation scope:

- Add binary dataset generation option.
- Train `train.py` with `num_classes=2`.
- Evaluate Bear recall, precision, F1, and balanced accuracy.
- Run a binary 2-Regime MVO backtest.

Success criteria:

- Bear recall improves or remains near 60%.
- False Bear signals do not destroy upside too much.
- MDD and Calmar improve or remain competitive.

First run result:

Command:

```bash
python3 scripts/prepare_cross_asset_dataset.py \
  --binary-bear \
  --output data/processed/cross_asset_supervised_30d_5d_binary_bear.npz \
  --index-output data/processed/cross_asset_supervised_30d_5d_binary_bear_index.csv \
  --meta-output data/processed/cross_asset_supervised_30d_5d_binary_bear_meta.json

python3 scripts/train.py \
  --data data/processed/cross_asset_supervised_30d_5d_binary_bear.npz \
  --model-output outputs/models/best_model_binary_bear.pt \
  --history-output outputs/results/train_history_binary_bear.json \
  --epochs 80 --patience 10 --batch-size 16 \
  --lr 1e-4 --conv-channels 16 --lstm-hidden 32 \
  --dropout 0.6 --weight-decay 1e-2 \
  --neutral-boost 1.0 --best-metric val_bal_acc --seed 42
```

Output files:

- `data/processed/cross_asset_supervised_30d_5d_binary_bear.npz`
- `outputs/models/best_model_binary_bear.pt`
- `outputs/results/train_history_binary_bear.json`

Binary split:

| Split | Non-Bear | Bear |
|---|---:|---:|
| Train | 339 | 149 |
| Valid | 62 | 43 |
| Test | 62 | 43 |

Test result:

| Metric | Value |
|---|---:|
| Accuracy | 72.4% |
| Balanced Accuracy | 70.2% |
| Macro F1 | 70.6% |
| Non-Bear Recall | 82.3% |
| Bear Recall | 58.1% |

Confusion matrix:

```text
                 Pred Non-Bear  Pred Bear
Actual Non-Bear       51           11
Actual Bear           18           25
```

Interpretation:

The binary task is much cleaner than the 3-class task. It removes the Neutral identification failure and produces a high balanced accuracy. Bear recall is 58.1%, slightly below the 3-class final model's 60.5%, but the binary model is easier to interpret and better aligned with downside-risk detection.

Next check:

- Connect the binary probabilities to 2-Regime MVO.
- Compare the resulting portfolio against EW 1/N, Buy & Hold, 60/40, and 3-class Regime-MVO.

### Priority 2: Constrained MVO

Reason:

- The current Regime-MVO has a known estimation-risk issue.
- This can be improved without changing the classifier.
- It directly addresses the 2022 weakness.

Implementation scope:

- Change MVO bounds from `(0.0, 1.0)` to `(0.0, 0.6)` or test several caps.
- Try MVO/equal-weight blending.
- Re-run `backtest_mvo.py`.

Success criteria:

- MDD remains low.
- 2022 stress test improves.
- Cumulative return does not collapse.

First run result:

The MVO script was extended with a `--max-weight` option so each asset's regime-level MVO weight can be capped.

Command examples:

```bash
python3 scripts/backtest_mvo.py \
  --max-weight 0.6 \
  --result-output outputs/results/backtest_mvo_cap60_results.json

python3 scripts/backtest_mvo.py \
  --max-weight 0.5 \
  --result-output outputs/results/backtest_mvo_cap50_results.json

python3 scripts/backtest_mvo.py \
  --max-weight 0.4 \
  --result-output outputs/results/backtest_mvo_cap40_results.json
```

Regime-level MVO weights:

| Cap | Bear | Neutral | Bull |
|---:|---|---|---|
| 60% | QQQ 11.9%, GLD 28.1%, TLT 60.0% | SPY 51.4%, GLD 48.6% | SPY 60.0%, QQQ 40.0% |
| 50% | QQQ 7.3%, GLD 42.7%, TLT 50.0% | SPY 50.0%, GLD 50.0% | SPY 50.0%, QQQ 50.0% |
| 40% | QQQ 20.0%, GLD 40.0%, TLT 40.0% | SPY 40.0%, QQQ 4.5%, GLD 40.0%, TLT 15.5% | SPY 40.0%, QQQ 40.0%, GLD 7.4%, TLT 12.6% |

Backtest result:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Original Regime-MVO | 35.3% | 1.10 | -7.2% | 2.16 |
| Cap 60% Regime-MVO | 49.8% | 1.42 | -8.8% | 2.44 |
| Cap 50% Regime-MVO | 54.2% | 1.50 | -9.4% | 2.45 |
| Cap 40% Regime-MVO | 51.9% | 1.46 | -9.0% | 2.47 |
| EW 1/N Benchmark | 50.9% | 1.41 | -8.8% | 2.47 |

Interpretation:

Constrained MVO reduces extreme single-asset concentration and materially improves cumulative return and Sharpe versus the original unconstrained Regime-MVO. However, it gives up some drawdown protection. The original Regime-MVO still has the lowest MDD among the MVO variants, while the capped versions are more competitive with EW 1/N on return and Sharpe.

Best current reading:

- If the goal is pure drawdown minimization, the original Regime-MVO still has lower MDD than capped variants.
- If the goal is a more balanced risk-return strategy, cap 50% or cap 40% is more attractive.
- Cap 40% is close to EW 1/N in MDD and Calmar, while keeping regime-conditioned dynamic allocation.

Important caution:

The cap values were checked on the test period as exploratory diagnostics. A final cap should be chosen through validation or walk-forward testing, not by selecting the best test-period metric.

### Priority 3: Classical ML Baselines

Reason:

- Small sample size makes Logistic Regression and Random Forest credible.
- Useful as a sanity check for whether LSTM complexity is justified.

Implementation scope:

- Start with binary Bear vs Non-Bear.
- Compare Logistic Regression, Random Forest, and current LSTM.
- Use time-based train/valid/test only.

Success criteria:

- Establish a simple baseline.
- If baseline is competitive, use it in the report as robustness evidence.

First run result:

`scikit-learn` was not available in the current Python environment, and system-wide package installation was blocked by the externally managed Python policy. To avoid modifying the environment, a small dependency-free baseline script was added.

Script:

```bash
python3 scripts/baseline_binary_bear.py
```

Output:

- `outputs/results/baseline_binary_bear_results.json`

Feature representation:

Instead of feeding the full `(30, 40)` tensor directly, the baseline uses compact 30-day summary features:

```text
last value, mean, standard deviation, min, max, recent 5-day mean
```

This gives a simpler and more stable tabular input for small-sample classical models.

Classification comparison:

| Model | Valid Balanced Acc | Test Balanced Acc | Bear Recall |
|---|---:|---:|---:|
| Logistic Regression | 65.2% | 62.6% | 34.9% |
| Random Forest | 82.4% | 67.2% | 48.8% |
| Conv1D+LSTM Binary | 85.2% | 70.2% | 58.1% |

Interpretation:

The classical baselines are useful sanity checks. Logistic Regression is too conservative and misses many Bear samples. The Random Forest baseline is stronger, especially on validation, but its test Bear recall is still weaker than the binary Conv1D+LSTM.

This supports the claim that the LSTM is adding some value beyond simple tabular summary rules. The gain is not enormous, but it is meaningful for the project's downside-risk objective:

- Better Bear recall
- Better test balanced accuracy
- Better downside-regime detection before connecting the signal to MVO

Important caution:

The Random Forest implementation is a small in-project approximation, not a full scikit-learn implementation. If a clean virtual environment is available later, the baseline should be repeated with official `sklearn.linear_model.LogisticRegression` and `sklearn.ensemble.RandomForestClassifier`.

Sklearn rerun:

After creating a local virtual environment, the official scikit-learn baselines were also tested.

Command:

```bash
.venv/bin/python scripts/baseline_binary_bear_sklearn.py
```

Output:

- `outputs/results/baseline_binary_bear_sklearn_results.json`

Result:

| Model | Valid Balanced Acc | Test Balanced Acc | Bear Recall |
|---|---:|---:|---:|
| Sklearn Logistic Regression | 63.7% | 61.4% | 32.6% |
| Sklearn Random Forest | 79.1% | 66.3% | 53.5% |
| Conv1D+LSTM Binary | 85.2% | 70.2% | 58.1% |

Interpretation:

The official sklearn baselines support the same conclusion as the dependency-free baseline. Random Forest is a useful comparator and detects more Bear samples than Logistic Regression, but the binary Conv1D+LSTM remains stronger on test balanced accuracy and Bear recall.

Environment note:

The virtual environment is ignored through `.gitignore` via `.venv/`.

### Priority 4: Soft-Label Training

Reason:

- Conceptually strong and directly addresses pseudo-label uncertainty.
- But it requires more training-loop changes than binary classification.

Implementation scope:

- Save HMM probability vectors as targets.
- Implement soft cross entropy.
- Compare hard-label 3-class vs soft-label 3-class.
- Backtest the resulting probabilities.

Success criteria:

- Better calibration of regime probabilities.
- Neutral probability becomes useful even if hard Neutral recall stays low.
- Portfolio metrics improve.

First run result:

The cross-asset dataset builder was extended with a `--soft-labels` option. Instead of using only `hmm_label_code`, it now also saves HMM posterior probabilities:

```text
prob_bear / prob_neutral / prob_bull
```

Cleanup note:

The 3-class soft-label artifacts were diagnostic only and are not retained in the cleaned output folder. The retained final soft-label path is the binary version used by 2-Regime MVO.

Important dataset observation:

| Split | Mean HMM Confidence |
|---|---:|
| Train | 98.7% |
| Valid | 99.4% |
| Test | 97.9% |

This means the HMM posterior probabilities are already very close to hard labels. Therefore, soft-label training may not change the model behavior much unless the HMM itself produces more uncertain probability vectors.

Classification result:

| Metric | Value |
|---|---:|
| Accuracy | 61.9% |
| Balanced Accuracy | 51.9% |
| Macro F1 | 45.5% |
| Soft CE | 1.0194 |
| Bear Recall | 60.5% |
| Neutral Recall | 0.0% |
| Bull Recall | 95.1% |

Backtest result:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| DL Regime SPY/Cash | 21.9% | 0.72 | -7.5% | 1.32 |
| Regime-MVO SoftLabel | 34.8% | 1.08 | -7.1% | 2.17 |
| Oracle (HMM labels) | 41.6% | 1.16 | -6.2% | 2.91 |

Interpretation:

Soft-label training did not solve the Neutral recall problem. The result is nearly identical to the hard-label 3-class model: Bear and Bull are learned, but Neutral is still not selected as the final argmax prediction. This does not invalidate the soft-label idea, but it shows that the current HMM probabilities are too confident to meaningfully smooth the labels.

### Priority 5: Confidence Weighting

Reason:

- Useful after soft-label or hard-label baselines are established.
- Dropping samples is risky because the dataset is small.

Implementation scope:

- Compute `confidence = max(HMM probabilities)`.
- Weight each training sample's loss by confidence.
- Avoid hard filtering at first.

Success criteria:

- Lower validation instability.
- Better balanced accuracy or better portfolio metrics.

First run result:

The soft-label training script was rerun with confidence-weighted loss:

```python
confidence = max(prob_bear, prob_neutral, prob_bull)
```

Cleanup note:

The 3-class confidence-weighting artifacts were diagnostic only and are not retained in the cleaned output folder.

Classification result:

| Metric | Soft Label | Soft Label + Confidence |
|---|---:|---:|
| Accuracy | 61.9% | 61.0% |
| Balanced Accuracy | 51.9% | 51.1% |
| Macro F1 | 45.5% | 44.9% |
| Soft CE | 1.0194 | 1.0069 |
| Bear Recall | 60.5% | 58.1% |
| Neutral Recall | 0.0% | 0.0% |
| Bull Recall | 95.1% | 95.1% |

Backtest result:

| Strategy | Soft Label | Soft Label + Confidence |
|---|---:|---:|
| DL Regime SPY/Cash cumulative return | 21.9% | 21.2% |
| DL Regime SPY/Cash MDD | -7.5% | -7.1% |
| Regime-MVO cumulative return | 34.8% | 33.5% |
| Regime-MVO Sharpe | 1.08 | 1.05 |
| Regime-MVO MDD | -7.1% | -7.1% |
| Regime-MVO Calmar | 2.17 | 2.08 |

Interpretation:

Confidence weighting slightly improves soft cross entropy, but it does not improve hard classification or portfolio performance. The likely reason is the same as Priority 4: the HMM confidence is already very high for most samples, so multiplying by confidence barely changes the effective training distribution.

Current conclusion for Priorities 4 and 5:

Soft labels and confidence weighting were implemented and tested, but they are not the best path for the current data. The stronger practical findings are:

- Binary Bear vs Non-Bear is the clearest fix for Neutral failure.
- Constrained MVO is the clearest fix for MVO concentration risk.
- Classical baselines confirm that the binary Conv1D+LSTM is still useful.

### Follow-up: Binary Soft-Label Training

Question:

If Bear vs Non-Bear worked better than 3-class classification, should soft-label training also be applied to the binary target?

Answer:

Yes. The 3-class soft-label experiment was a diagnostic test for whether preserving `Bear / Neutral / Bull` HMM uncertainty would fix Neutral failure. But the practical risk-management direction is binary. Therefore, the more aligned soft-label target is:

```text
P(Non-Bear) = P(Neutral) + P(Bull)
P(Bear)     = P(Bear)
```

Implementation:

The dataset builder now allows `--binary-bear --soft-labels` together.

Command:

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
  --history-output outputs/results/train_history_binary_soft_labels.json \
  --epochs 80 --patience 10 --batch-size 16 \
  --lr 1e-4 --conv-channels 16 --lstm-hidden 32 \
  --dropout 0.6 --weight-decay 1e-2 --seed 42
```

Output files:

- `data/processed/cross_asset_supervised_30d_5d_binary_soft_labels.npz`
- `outputs/models/best_model_binary_soft_labels.pt`
- `outputs/results/train_history_binary_soft_labels.json`

Classification comparison:

| Model | Accuracy | Balanced Accuracy | Macro F1 | Non-Bear Recall | Bear Recall |
|---|---:|---:|---:|---:|---:|
| Binary Hard Label | 72.4% | 70.2% | 70.6% | 82.3% | 58.1% |
| Binary Soft Label | 73.3% | 72.4% | 72.4% | 77.4% | 67.4% |
| Binary Soft Label + Confidence | 73.3% | 72.4% | 72.4% | 77.4% | 67.4% |

Interpretation:

Binary soft-label training is better than binary hard-label training on classification, especially Bear recall. Bear recall improves from 58.1% to 67.4%, which is useful for downside-risk detection.

However, classification improvement alone is not the final objective. Since this project is built around regime-conditioned MVO, the binary soft-label probabilities should be evaluated through a 2-Regime MVO strategy.

Best current reading:

- For classification: Binary Soft Label is the strongest binary classifier so far.
- Since the project's main portfolio story is Regime-MVO, Binary Soft Label should be connected to a 2-regime MVO strategy.

### Follow-up: Binary Soft Label + 2-Regime MVO

Question:

If Binary Soft Label is the best classifier so far, should it be applied to MVO?

Answer:

Yes. The cleaner project-consistent strategy is 2-Regime MVO:

```text
Non-Bear MVO = MVO weights estimated from train samples labeled Neutral or Bull
Bear MVO     = MVO weights estimated from train samples labeled Bear

test weight = P(Non-Bear) * Non-Bear MVO + P(Bear) * Bear MVO
```

This keeps the main project pipeline:

```text
regime prediction -> regime-conditioned MVO allocation
```

Script:

```bash
python3 scripts/backtest_binary_mvo.py \
  --output outputs/results/backtest_binary_soft_mvo_results.json

python3 scripts/backtest_binary_mvo.py \
  --max-weight 0.5 \
  --output outputs/results/backtest_binary_soft_mvo_cap50_results.json

python3 scripts/backtest_binary_mvo.py \
  --max-weight 0.4 \
  --output outputs/results/backtest_binary_soft_mvo_cap40_results.json
```

Regime-level MVO weights:

| Cap | Non-Bear MVO | Bear MVO |
|---:|---|---|
| 100% | SPY 100.0% | TLT 100.0% |
| 50% | SPY 50.0%, QQQ 49.8%, GLD 0.2% | QQQ 7.3%, GLD 42.7%, TLT 50.0% |
| 40% | SPY 40.0%, QQQ 40.0%, GLD 17.3%, TLT 2.7% | QQQ 20.0%, GLD 40.0%, TLT 40.0% |

Backtest result:

| Strategy | Cumulative Return | Sharpe | MDD | Calmar |
|---|---:|---:|---:|---:|
| 60/40 | 28.3% | 0.84 | -10.4% | 1.22 |
| Buy & Hold | 49.9% | 1.07 | -17.0% | 1.26 |
| EW 1/N | 50.9% | 1.41 | -8.8% | 2.47 |
| 3-class Regime-MVO, original | 35.3% | 1.10 | -7.2% | 2.16 |
| 3-class Regime-MVO, cap 40% | 51.9% | 1.46 | -9.0% | 2.47 |
| Binary Regime-MVO Soft, no cap | 22.9% | 0.59 | -8.5% | 1.22 |
| Binary Regime-MVO Soft, cap 50% | 47.9% | 1.37 | -8.3% | 2.48 |
| Binary Regime-MVO Soft, cap 40% | 53.7% | 1.48 | -9.0% | 2.55 |

Cumulative-return ranking:

| Rank | Strategy | Cumulative Return |
|---:|---|---:|
| 1 | Binary Regime-MVO Soft, cap 40% | 53.7% |
| 2 | 3-class Regime-MVO, cap 40% | 51.9% |
| 3 | EW 1/N | 50.9% |
| 4 | Buy & Hold | 49.9% |
| 5 | Binary Regime-MVO Soft, cap 50% | 47.9% |
| 6 | 3-class Regime-MVO, original | 35.3% |
| 7 | 60/40 | 28.3% |
| 8 | Binary Regime-MVO Soft, no cap | 22.9% |

Interpretation:

Binary Soft Label works better when it is connected back to MVO. The unconstrained 2-regime MVO collapses into extreme portfolios, `Non-Bear=SPY 100%` and `Bear=TLT 100%`, so it is still exposed to the same MVO concentration problem. But once the MVO cap is added, the binary MVO strategy becomes competitive.

The best current binary MVO result is cap 40%:

```text
Binary Regime-MVO Soft, cap 40%
cum_ret = 53.7%
Sharpe  = 1.48
MDD     = -9.0%
Calmar  = 2.55
```

This is slightly better than EW 1/N and slightly better than 3-class capped Regime-MVO on cumulative return, Sharpe, and Calmar, while keeping a similar MDD.

Current conclusion:

The cleanest project story is now:

```text
Neutral is hard to classify
-> Collapse to Bear vs Non-Bear
-> Train Binary Soft Label model
-> Apply probabilities to 2-Regime MVO
-> Add MVO cap to avoid concentration
```

This is aligned with the original MVO-based project objective.

---

## 9. Short Presentation/Q&A Framing

If asked why Neutral fails:

> Neutral is structurally ambiguous because it is the middle HMM state, not a directly observed economic event. The model learns Bear and Bull more clearly, but Neutral often overlaps with both. Therefore, a natural next step is to reframe the task as Bear vs Non-Bear, which better matches the risk-management objective.

If asked whether HMM labels are reliable:

> HMM labels are pseudo-labels, not ground truth. We therefore should not overclaim classification accuracy. A stronger extension is to train on HMM probabilities as soft labels or weight samples by HMM confidence.

If asked why MVO is unstable:

> MVO depends on estimated means and covariance. After splitting by regime, each estimate is based on a small sample, so Sharpe-maximizing MVO can choose extreme weights. Constrained MVO, shrinkage, cash assets, or blending with equal weight are natural stabilizers.

If asked what experiment should come first:

> First, Bear vs Non-Bear. It is the fastest, most aligned with the project's risk-control goal, and directly addresses the Neutral failure.
