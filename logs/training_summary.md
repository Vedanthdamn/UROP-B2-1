# Federated Training Experiment Summary

**Generated:** 2026-01-31 04:40:28

---

## Experiment Configuration

- **Strategy:** fedavg
- **Number of Clients:** 5
- **Number of Rounds:** 5
- **Model:** LSTM (PRIMARY)
- **Data Partitioning:** Non-IID
- **Differential Privacy:** Disabled
- **Random Seed:** 42

## Per-Round Training Metrics

| Round | Global Loss | Global Accuracy | Participating Clients | Total Samples |
|-------|-------------|-----------------|----------------------|---------------|
| 1 | 0.6858 | 0.5504 | 5 | 238 |
| 2 | 0.6771 | 0.6176 | 5 | 238 |
| 3 | 0.6648 | 0.6639 | 5 | 238 |
| 4 | 0.6559 | 0.6849 | 5 | 238 |
| 5 | 0.6418 | 0.7563 | 5 | 238 |

## Training Summary

- **Initial Loss:** 0.6858
- **Final Loss:** 0.6418
- **Loss Reduction:** 0.0440
- **Average Loss:** 0.6651

- **Initial Accuracy:** 0.5504
- **Final Accuracy:** 0.7563
- **Accuracy Improvement:** 0.2059
- **Average Accuracy:** 0.6546

## Privacy Guarantees

- ✓ Raw patient data remained on hospital clients
- ✓ Only model weights were shared with server
- ✓ Server never accessed individual patient records
- ✓ All client updates treated as privacy-protected

## Client Participation Summary

- **Total Clients:** 5
- **Data Partitioning:** Non-IID (realistic hospital data distribution)
- **Participation Rate:** 100% (all clients participated in each round)
