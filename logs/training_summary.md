# Federated Training Experiment Summary

**Generated:** 2026-01-31 04:33:21

---

## Experiment Configuration

- **Strategy:** fedprox
- **Number of Clients:** 5
- **Number of Rounds:** 2
- **Model:** LSTM (PRIMARY)
- **Data Partitioning:** Non-IID
- **Differential Privacy:** Disabled
- **Proximal Mu:** 0.1
- **Random Seed:** 42

## Per-Round Training Metrics

| Round | Global Loss | Global Accuracy | Participating Clients | Total Samples |
|-------|-------------|-----------------|----------------------|---------------|
| 1 | 0.6924 | 0.5294 | 5 | 238 |
| 2 | 0.6824 | 0.5588 | 5 | 238 |

## Training Summary

- **Initial Loss:** 0.6924
- **Final Loss:** 0.6824
- **Loss Improvement:** 0.0100
- **Average Loss:** 0.6874

- **Initial Accuracy:** 0.5294
- **Final Accuracy:** 0.5588
- **Accuracy Improvement:** 0.0294
- **Average Accuracy:** 0.5441

## Privacy Guarantees

- ✓ Raw patient data remained on hospital clients
- ✓ Only model weights were shared with server
- ✓ Server never accessed individual patient records
- ✓ All client updates treated as privacy-protected

## Client Participation Summary

- **Total Clients:** 5
- **Data Partitioning:** Non-IID (realistic hospital data distribution)
- **Participation Rate:** 100% (all clients participated in each round)
