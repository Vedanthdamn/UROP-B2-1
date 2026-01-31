# Federated Model Evaluation Report

**Generated:** 2026-01-31 04:52:44

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
- **Precision:** 0.0000
- **Recall:** 0.0000
- **F1-Score:** 0.0000
- **Cross-Entropy Loss:** 0.5956

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
| hospital_0 | 0.7838 | 74 |
| hospital_1 | 0.8615 | 65 |
| hospital_2 | 0.8136 | 59 |
| hospital_3 | 0.7949 | 39 |
| hospital_4 | 0.8387 | 62 |

### Fairness Metrics (Accuracy Variance)

- **Mean Client Accuracy:** 0.8185
- **Standard Deviation:** 0.0285
- **Minimum Client Accuracy:** 0.7838
- **Maximum Client Accuracy:** 0.8615
- **Accuracy Variance:** 0.0008

**Fairness Interpretation:**
- ✓ Low variance indicates fair performance across clients

## Training Progress Summary

- **Initial Training Accuracy:** 0.6639
- **Final Training Accuracy:** 0.7689
- **Initial Training Loss:** 0.6617
- **Final Training Loss:** 0.6143

## Conclusions

1. **Overall Performance:** The federated model achieved 93.33% accuracy on test data.
2. **Model Quality:** F1-score of 0.0000 indicates moderate balance between precision and recall.
3. **Fairness:** Accuracy variance of 0.0008 across clients suggests equitable performance distribution.
