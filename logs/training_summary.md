# Federated Training Experiment Summary

**Generated:** 2026-01-31 04:38:04

---

## Experiment Configuration

- **Strategy:** fedavg
- **Number of Clients:** 5
- **Number of Rounds:** 2
- **Model:** LSTM (PRIMARY)
- **Data Partitioning:** Non-IID
- **Differential Privacy:** Enabled
  - Epsilon (ε): 1.0
  - Delta (δ): 1e-05
  - L2 Norm Clip: 1.0
- **Random Seed:** 42

## Per-Round Training Metrics

| Round | Global Loss | Global Accuracy | Participating Clients | Total Samples |
|-------|-------------|-----------------|----------------------|---------------|
| 1 | 0.6964 | 0.4790 | 5 | 238 |
| 2 | 2.3890 | 0.5294 | 5 | 238 |

## Training Summary

- **Initial Loss:** 0.6964
- **Final Loss:** 2.3890
- **Loss Improvement:** -1.6926
- **Average Loss:** 1.5427

- **Initial Accuracy:** 0.4790
- **Final Accuracy:** 0.5294
- **Accuracy Improvement:** 0.0504
- **Average Accuracy:** 0.5042

## Privacy Guarantees

- ✓ Raw patient data remained on hospital clients
- ✓ Only model weights were shared with server
- ✓ Server never accessed individual patient records
- ✓ All client updates treated as privacy-protected
- ✓ Differential privacy enabled with (ε=1.0, δ=1e-05)

## Client Participation Summary

- **Total Clients:** 5
- **Data Partitioning:** Non-IID (realistic hospital data distribution)
- **Participation Rate:** 100% (all clients participated in each round)
