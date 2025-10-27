# History-Agnostic LIB SoH and RUL Estimation using BNN

This repository contains the official code for the paper:

"History-Agnostic Lithium-Ion Batteries State of Health and Remaining Useful Life Estimation Using Fusion-Based Bayesian Neural Network" 

💡 Proposed Methodology
The proposed framework operates in two stages: Offline Training and Online Prediction.

Feature Extraction: Ten interpretable Health Indicators (HIs) are extracted from a single charge-discharge cycle's voltage profile. This includes four novel HIs derived from the voltage-time derivative (dV/dt).

Model Training (Offline): Two separate Bayesian Neural Networks (BNNs) are trained on the extracted HI dataset—one for SoH estimation and one for direct RUL prediction. The models are trained using the Bayes by Backpropagation (BBB) algorithm.

Prediction (Online): The trained models can take the 10 HIs from a single cycle of a new battery (with no known history) and instantly generate probabilistic SoH and RUL predictions with uncertainty bounds.
