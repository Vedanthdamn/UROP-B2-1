# Client Partition Summary

## Overview
- **Number of Clients**: 5
- **Total Samples**: 299
- **Random Seed**: 42
- **Partitioning Strategy**: Non-IID (unequal sizes, different class distributions)

## Client Statistics

| Client ID | Number of Samples | Class 0 Count | Class 0 % | Class 1 Count | Class 1 % |
|-----------|-------------------|---------------|-----------|---------------|-----------|
| Client 0 | 74 | 55 | 74.32% | 19 | 25.68% |
| Client 1 | 65 | 39 | 60.00% | 26 | 40.00% |
| Client 2 | 59 | 29 | 49.15% | 30 | 50.85% |
| Client 3 | 39 | 18 | 46.15% | 21 | 53.85% |
| Client 4 | 62 | 62 | 100.00% | 0 | 0.00% |

## Detailed Client Distributions

### Client 0

- **Total Samples**: 74
- **Percentage of Total Dataset**: 24.75%

**Class Distribution:**

- Class 0: 55 samples (74.32%)
- Class 1: 19 samples (25.68%)

### Client 1

- **Total Samples**: 65
- **Percentage of Total Dataset**: 21.74%

**Class Distribution:**

- Class 0: 39 samples (60.00%)
- Class 1: 26 samples (40.00%)

### Client 2

- **Total Samples**: 59
- **Percentage of Total Dataset**: 19.73%

**Class Distribution:**

- Class 0: 29 samples (49.15%)
- Class 1: 30 samples (50.85%)

### Client 3

- **Total Samples**: 39
- **Percentage of Total Dataset**: 13.04%

**Class Distribution:**

- Class 0: 18 samples (46.15%)
- Class 1: 21 samples (53.85%)

### Client 4

- **Total Samples**: 62
- **Percentage of Total Dataset**: 20.74%

**Class Distribution:**

- Class 0: 62 samples (100.00%)

## Non-IID Characteristics

### Unequal Sample Sizes

- Minimum samples per client: 39
- Maximum samples per client: 74
- Average samples per client: 59.8
- Standard deviation: 11.5

### Different Class Distributions

- Class 0 proportion range: 46.15% - 100.00%
- Class 0 proportion std dev: 19.69%

This demonstrates clear non-IID characteristics with both unequal sample sizes and varying class distributions across clients.

## Data Integrity

✓ **No overlap between clients**: Each sample appears in exactly one client dataset

✓ **Total samples match**: 299 samples partitioned across 5 clients

## Reproducibility

The partition is deterministic and reproducible using random seed `42`.
Running the partition with the same seed will always produce the same results.

## Notes

- Each client represents a federated hospital with its own data characteristics
- Different sample sizes simulate real-world scenarios where hospitals have different patient volumes
- Different class distributions simulate different patient populations across hospitals
- No samples overlap between clients (proper partitioning without duplication)
- The last client receives all remaining samples, which may result in extreme class distributions
  in some cases. This simulates real-world scenarios where some hospitals may have highly
  imbalanced patient populations.
