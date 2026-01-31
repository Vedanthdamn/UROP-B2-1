# Privacy-Utility Analysis for Federated Learning

**Generated:** 2026-01-31 05:26:44

---

## Analysis Overview

This report analyzes the trade-off between privacy and model utility 
in federated learning with differential privacy. By varying the privacy 
budget (epsilon), we measure the impact on model accuracy and loss.


## Experiment Configuration

- **Number of Clients:** 5
- **Number of Rounds:** 10
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
| 0 | None | No differential privacy (baseline) |
| 0.5 | Strong | High privacy protection, potential utility loss |
| 1.0 | Moderate | Balanced privacy-utility trade-off |
| 2.0 | Moderate | Balanced privacy-utility trade-off |
| 5.0 | Relaxed | Lower privacy protection, higher utility |

## Results Summary

| Epsilon (ε) | Final Accuracy | Final Loss | Client Accuracy Variance |
|-------------|----------------|------------|-------------------------|
| 0 | 0.8655 | 0.5666 | 0.026202 |
| 0.5 | 0.5084 | 50.2011 | 0.031537 |
| 1.0 | 0.4916 | 22.3805 | 0.003379 |
| 2.0 | 0.5126 | 12.0306 | 0.007646 |
| 5.0 | 0.5084 | 4.4987 | 0.006825 |

## Key Findings

### Privacy-Utility Trade-off

**Baseline (No DP):** Accuracy = 0.8655, Loss = 0.5666

**ε = 0.5:**
- Accuracy: 0.5084 (-0.3571, -41.26%)
- Loss: 50.2011 (+49.6345)

**ε = 1.0:**
- Accuracy: 0.4916 (-0.3739, -43.20%)
- Loss: 22.3805 (+21.8140)

**ε = 2.0:**
- Accuracy: 0.5126 (-0.3529, -40.78%)
- Loss: 12.0306 (+11.4641)

**ε = 5.0:**
- Accuracy: 0.5084 (-0.3571, -41.26%)
- Loss: 4.4987 (+3.9321)

### Fairness Analysis

Client accuracy variance measures how fairly the model performs across 
different hospitals. Lower variance indicates more equitable performance.

- **Lowest variance:** ε = 1.0 (variance = 0.003379)
- **Highest variance:** ε = 0.5 (variance = 0.031537)

## Recommendations

Based on the privacy-utility analysis:

1. **For maximum utility:** Use higher epsilon (ε ≥ 5.0) or no DP if privacy is not a concern
2. **For balanced trade-off:** Use moderate epsilon (ε = 1.0 - 2.0)
3. **For strong privacy:** Use low epsilon (ε ≤ 0.5), accepting potential utility loss
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
