# Federated Model Evaluation Report

**Generated:** 2026-01-31 04:54:29

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

- **Accuracy:** 0.9333 (93.33%)
- **Precision (Death class):** 0.0000
- **Recall (Death class):** 0.0000
- **F1-Score (Death class):** 0.0000
- **Precision (Weighted):** 0.9017
- **Recall (Weighted):** 0.9333
- **F1-Score (Weighted):** 0.9172
- **Cross-Entropy Loss:** 0.5878

### Data Distribution

- **Test Set Distribution:** Survived=57, Death=3
- **Predicted Distribution:** Survived=59, Death=1

⚠️ **Note:** The model shows class imbalance issues. It's predicting predominantly one class.
This is common in medical datasets where one outcome is more frequent than the other.

### Confusion Matrix

```
                 Predicted
              Survived  Death
Actual Survived     56       1
       Death         3       0
```

See `confusion_matrix.png` for visualization.

## Federated-Specific Metrics

### Client-Level Performance

| Client ID | Accuracy | Samples |
|-----------|----------|---------|
| hospital_0 | 0.7568 | 74 |
| hospital_1 | 0.7692 | 65 |
| hospital_2 | 0.7797 | 59 |
| hospital_3 | 0.7692 | 39 |
| hospital_4 | 0.8226 | 62 |

### Fairness Metrics (Accuracy Variance)

- **Mean Client Accuracy:** 0.7795
- **Standard Deviation:** 0.0227
- **Minimum Client Accuracy:** 0.7568
- **Maximum Client Accuracy:** 0.8226
- **Accuracy Variance:** 0.0005

**Fairness Interpretation:**
- ✓ Low variance indicates fair performance across clients

## Training Progress Summary

- **Initial Training Accuracy:** 0.6639
- **Final Training Accuracy:** 0.7689
- **Initial Training Loss:** 0.6617
- **Final Training Loss:** 0.6143

## Conclusions

1. **Overall Performance:** The federated model achieved 93.33% accuracy on test data.
2. **Model Quality:** Weighted F1-score of 0.9172 indicates good overall performance accounting for class imbalance.
3. **Fairness:** Accuracy variance of 0.0005 across clients suggests equitable performance distribution.
4. **Client Consistency:** Standard deviation of 0.0227 across clients indicates high consistency in model performance across different hospital datasets.
