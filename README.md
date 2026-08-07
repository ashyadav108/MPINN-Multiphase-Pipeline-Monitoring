# MPINN: Multi-Head Physics-Informed Neural Network for Multiphase Pipeline Monitoring

> **Developed as part of M.Tech research in Communication System Engineering at NIT Jamshedpur.** 
> This repository contains the code and evaluation framework for an advanced anomaly detection node designed for Industrial IoT (IIoT) sensor networks.

## Overview
Real-time leak detection in multiphase subsea and industrial pipelines is a challenging problem due to highly nonlinear flow interactions, sparse sensor availability, and the limitations of traditional computational fluid dynamics (CFD) models. 

This project introduces a **Multi-Head Physics-Informed Neural Network (MPINN)** that seamlessly blends data-driven deep learning with the fundamental laws of fluid mechanics. By acting as a highly efficient edge-computing model, this network simultaneously classifies the existence of a leak, estimates the leak diameter down to the millimeter, and localizes the rupture across a 6,500-meter pipeline with bounded uncertainty.

---

## Novel Contributions (Differences from Baseline)
This repository extends and optimizes standard MPINN architectures by introducing four critical engineering improvements to solve gradient instability and metric distortion:

1. **The Loss Function Mathematical Form:** Replaced standard Mean Squared Error (MSE) with L1 Loss (Mean Absolute Error) for dimensional regression. Squaring millimeter-scale leak sizes (e.g., 0.005m) previously resulted in infinitesimally small penalties that the optimizer ignored; L1 Loss forces the network to respect micro-scale physical dimensions.
2. **Custom Target Normalization:** Implemented a Sigmoid-based scaling trick for spatial bounding. The network predicts a percentage ($0.0$ to $1.0$) which is mathematically scaled to the maximum pipe length ($6500.0$ meters), preventing the network from struggling to output raw high-magnitude values and eliminating severe gradient explosions.
3. **Anti-Bullying Loss Weighting:** Standard physics penalties (which calculate raw mass and pressure differences) naturally generate massive loss values that crush smaller regression penalties. This architecture introduces a micro-scaled composite loss function with a specific equilibrium weight ($W_{size} = 5000.0$) so the optimizer balances physical laws without losing millimeter-level size accuracy.
4. **Metrics Reality and Dataset Variance:** Rather than relying solely on $R^2$ scores—which are mathematically distorted by the low spatial variance (tight clustering) of anomalies in the dataset—this framework reports absolute engineering metrics (MAE and RMSE) for transparent real-world reliability.

---

## Explainability & Uncertainty Quantification
*   **Explainable AI (XAI):** Integrated with SHAP (SHapley Additive exPlanations) to prove the model's reliance on true fluid dynamics (e.g., Discharge Coefficient $C_d$, Inlet/Outlet Pressure differentials) rather than purely memorizing data patterns.
*   **Uncertainty Quantification:** Utilizes Monte Carlo Dropout to generate 95% Confidence Intervals (CI) for all regressions, allowing maintenance teams to rely on bounded search radii (aleatoric and epistemic uncertainty) rather than raw point estimates.

---

## Repository Structure

*   `model.py`: Defines the MPINN architecture, the custom task heads, and the composite Physics-Informed Loss function ($L_{total}$).
*   `dataset.py`: Handles data ingestion, SMOTE for class balancing, and standard Z-score scaling.
*   `train.py`: The main training loop utilizing the Adam optimizer, Cosine Annealing learning rate schedules, and gradient clipping.
*   `calculate_metrics.py`: Generates the quantitative performance evaluation table.
*   `evaluate.py`: Force-evaluates active leak samples, runs Monte Carlo Dropout for uncertainty bounds, and calculates SHAP values.
*   `paper_workflow.py`: Generates publication-grade, high-DPI parity plots and SHAP feature importance graphs for academic manuscripts.

---

## Usage Instructions

### 1. Training the Model
To train the MPINN from scratch and save the optimized weights (`trained_mpinn.pth`):
```bash
python train.py