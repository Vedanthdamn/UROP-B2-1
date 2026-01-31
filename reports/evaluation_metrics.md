# Federated Model Evaluation Report

**Generated:** 2026-01-31 04:58:17

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
- **Cross-Entropy Loss:** 0.6148

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
| hospital_0 | 0.7703 | 74 |
| hospital_1 | 0.7846 | 65 |
| hospital_2 | 0.7119 | 59 |
| hospital_3 | 0.5897 | 39 |
| hospital_4 | 0.8387 | 62 |

### Fairness Metrics (Accuracy Variance)

- **Mean Client Accuracy:** 0.7390
- **Standard Deviation:** 0.0849
- **Minimum Client Accuracy:** 0.5897
- **Maximum Client Accuracy:** 0.8387
- **Accuracy Variance:** 0.0072

**Fairness Interpretation:**
- ~ Moderate variance indicates some performance disparity

## Training Summary

## Conclusions

1. **Overall Performance:** The federated model achieved 90.00% accuracy on test data.
2. **Model Quality:** Weighted F1-score of 0.9116 indicates good overall performance accounting for class imbalance.
3. **Fairness:** Accuracy variance of 0.0072 across clients suggests equitable performance distribution.
4. **Client Consistency:** Standard deviation of 0.0849 across clients indicates moderate consistency in model performance across different hospital datasets.
