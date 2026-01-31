# Federated Model Evaluation Report

**Generated:** 2026-01-31 05:00:14

---

## Experiment Configuration

- **Strategy:** fedavg
- **Number of Clients:** 5
- **Number of Rounds:** 5
- **Model:** LSTM (PRIMARY)
- **Data Partitioning:** Non-IID
- **Random Seed:** 42

## Standard Evaluation Metrics

### Classification Performance

- **Accuracy:** 0.9000 (90.00%)
- **Precision (Death class):** 0.2000
- **Recall (Death class):** 0.3333
- **F1-Score (Death class):** 0.2500
- **Precision (Weighted):** 0.9255
- **Recall (Weighted):** 0.9000
- **F1-Score (Weighted):** 0.9116
- **Cross-Entropy Loss:** 0.6025

### Data Distribution

- **Test Set Distribution:** Survived=57, Death=3
- **Predicted Distribution:** Survived=55, Death=5

### Confusion Matrix

```
                 Predicted
              Survived  Death
Actual Survived     53       4
       Death         2       1
```

See `confusion_matrix.png` for visualization.

## Federated-Specific Metrics

### Client-Level Performance

| Client ID | Accuracy | Samples |
|-----------|----------|---------|
| hospital_0 | 0.7568 | 74 |
| hospital_1 | 0.7538 | 65 |
| hospital_2 | 0.7627 | 59 |
| hospital_3 | 0.6410 | 39 |
| hospital_4 | 0.7581 | 62 |

### Fairness Metrics (Accuracy Variance)

- **Mean Client Accuracy:** 0.7345
- **Standard Deviation:** 0.0468
- **Minimum Client Accuracy:** 0.6410
- **Maximum Client Accuracy:** 0.7627
- **Accuracy Variance:** 0.0022

**Fairness Interpretation:**
- ✓ Low variance indicates fair performance across clients

## Training Summary

- **Initial Training Accuracy:** 0.6639
- **Final Training Accuracy:** 0.7689
- **Accuracy Improvement:** 0.1050
- **Initial Training Loss:** 0.6617
- **Final Training Loss:** 0.6143
- **Loss Reduction:** 0.0474

## Conclusions

1. **Overall Performance:** The federated model achieved 90.00% accuracy on test data.
2. **Model Quality:** Weighted F1-score of 0.9116 indicates good overall performance accounting for class imbalance.
3. **Fairness:** Accuracy variance of 0.0022 across clients suggests equitable performance distribution.
4. **Client Consistency:** Standard deviation of 0.0468 across clients indicates high consistency in model performance across different hospital datasets.
