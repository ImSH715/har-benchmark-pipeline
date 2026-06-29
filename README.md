# har-benchmark-pipeline# HAR Benchmark Pipeline

A config-driven preprocessing pipeline that harmonises heterogeneous Human Activity Recognition (HAR) datasets into a unified, ML-ready benchmark for 6-axis lower-back IMU sensors.

## Overview

- **4 Datasets:** UCI HAR, KU-HAR, HARTH, RealDISP
- **Output format:** 2.56 s sliding windows (128 samples @ 50 Hz)
- **Dual outputs:** DL-ready tensors `(N, 128, 6)` & ML-ready feature vectors `(N, 54)`
- **Extensible:** JSON config enables plug-and-play integration of new clinical cohorts (Parkinson's, CP, PSP)

## Pipeline Architecture

1. **Data Ingestion** — Load raw IMU logs / subsamples
2. **Harmonisation** — Standardise sampling rates, axes, and label schemas
3. **Windowing** — 50% overlap sliding windows (128 samples)
4. **Quality Control** — Outlier clipping (Acc ±157 m/s², Gyr ±34.9 rad/s)
5. **Feature Extraction** — Statistical features (mean, std, RMS, skew, kurtosis)
6. **Train/Val/Test Split** — Stratified 70/15/15 split with metadata retention
