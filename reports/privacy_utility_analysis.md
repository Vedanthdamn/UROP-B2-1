# Privacy-Utility Analysis for Federated Learning

**Generated:** 2026-01-31 05:10:18

---

## Analysis Overview

This report analyzes the trade-off between privacy and model utility 
in federated learning with differential privacy. By varying the privacy 
budget (epsilon), we measure the impact on model accuracy and loss.


## Experiment Configuration

- **Number of Clients:** 3
- **Number of Rounds:** 3
- **Model Architecture:** LSTM (PRIMARY)
- **Data Partitioning:** Non-IID
- **Strategy:** FedAvg
- **Fixed Parameters:**
  - Delta (δ): 1e-5
  - L2 Norm Clip: 1.0
  - Epochs per Round: 5
  - Batch Size: 32
- **Random Seed:** 42

## Privacy Budgets Tested

| Epsilon (ε) | Privacy Level | Description |
|-------------|---------------|-------------|
| 0.0 | None | No differential privacy (baseline) |
| 1.0 | Moderate | Balanced privacy-utility trade-off |

## Results Summary

| Epsilon (ε) | Final Accuracy | Final Loss | Client Accuracy Variance |
|-------------|----------------|------------|-------------------------|
| 0.0 | 0.7448 | 0.6243 | 0.008587 |
| 1.0 | 0.4059 | 7.5986 | 0.028258 |

## Key Findings

### Privacy-Utility Trade-off

**Baseline (No DP):** Accuracy = 0.7448, Loss = 0.6243

**ε = 1.0:**
- Accuracy: 0.4059 (-0.3389, -45.51%)
- Loss: 7.5986 (+6.9743)

### Fairness Analysis

Client accuracy variance measures how fairly the model performs across 
different hospitals. Lower variance indicates more equitable performance.

- **Lowest variance:** ε = 0.0 (variance = 0.008587)
- **Highest variance:** ε = 1.0 (variance = 0.028258)

## Recommendations

Based on the privacy-utility analysis:

1. **For maximum utility:** Use higher epsilon (ε ≥ 3.0) or no DP if privacy is not a concern
2. **For balanced trade-off:** Use moderate epsilon (ε = 1.0 - 2.0)
3. **For strong privacy:** Use low epsilon (ε < 1.0), accepting potential utility loss
4. **Consider fairness:** Choose epsilon that minimizes client accuracy variance

## Visualizations

See the accompanying plots for visual analysis:

- `accuracy_vs_epsilon.png`: Shows how accuracy changes with privacy budget
- `loss_vs_epsilon.png`: Shows how loss changes with privacy budget

## Privacy Guarantees

All experiments with ε > 0 provide (ε, δ)-differential privacy guarantees where:

- **ε (epsilon):** Privacy budget - lower values provide stronger privacy
- **δ (delta):** Fixed at 1e-5 for all experiments
- **Mechanism:** Gradient clipping + Gaussian noise addition

**Note:** These privacy guarantees apply to model updates shared with the server. 
Raw patient data never leaves the hospital clients.
