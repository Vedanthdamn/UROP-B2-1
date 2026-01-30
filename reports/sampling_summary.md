# Dataset Sampling Summary

## Overview
- **Random Seed**: 42
- **Sample Size**: 299 records
- **Original Dataset Size**: 299 records
- **Sampling Method**: Stratified sampling (preserving class distribution)

## Class Distribution

### Original Dataset

- Class 0: 203 records (67.89%)
- Class 1: 96 records (32.11%)

### Sampled Dataset

- Class 0: 203 records (67.89%)
- Class 1: 96 records (32.11%)

## Distribution Preservation

✓ **Class distribution successfully preserved** (within 5% tolerance)

## Reproducibility

The sampling operation is deterministic and reproducible using random seed `42`.
Running the sampling with the same seed will always produce the same results.

## Notes

- The original dataset is preserved and not modified.
- Total records in original dataset: 299
- Total records in sampled dataset: 299
- Sampling uses stratified approach to maintain class balance.
