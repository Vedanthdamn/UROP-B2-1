# Federated Training Experiment Summary

**Generated:** 2026-01-31 04:44:24

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
| 1 | 0.6617 | 0.6639 | 5 | 238 |
| 2 | 0.6535 | 0.6975 | 5 | 238 |
| 3 | 0.6396 | 0.7689 | 5 | 238 |
| 4 | 0.6212 | 0.7941 | 5 | 238 |
| 5 | 0.6143 | 0.7689 | 5 | 238 |

## Training Summary

- **Initial Loss:** 0.6617
- **Final Loss:** 0.6143
- **Loss Reduction:** 0.0474
- **Average Loss:** 0.6380

- **Initial Accuracy:** 0.6639
- **Final Accuracy:** 0.7689
- **Accuracy Improvement:** 0.1050
- **Average Accuracy:** 0.7387

## Privacy Guarantees

- ✓ Raw patient data remained on hospital clients
- ✓ Only model weights were shared with server
- ✓ Server never accessed individual patient records
- ✓ All client updates treated as privacy-protected

## Client Participation Summary

- **Total Clients:** 5
- **Data Partitioning:** Non-IID (realistic hospital data distribution)
- **Participation Rate:** 100% (all clients participated in each round)
