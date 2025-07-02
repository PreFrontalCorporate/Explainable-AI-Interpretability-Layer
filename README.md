# Financial AI Explainability Module

This repository provides a practical module for **Explainable AI (XAI)** in financial machine learning models, focusing on credit risk assessment or approval prediction tasks. It includes a full example integrating SHAP (SHapley Additive exPlanations) to analyze feature contributions in a financial classification model. 

## 🚀 Motivation

In financial applications, model interpretability is critical for:

- Ensuring **trust and transparency** for regulators, auditors, and users.
- Understanding **why** a model approves or rejects an applicant.
- Avoiding hidden biases that could lead to unfair or discriminatory decisions.

Traditional ML models often act as black boxes, making it difficult to understand individual predictions or feature importance. **SHAP** offers a theoretically grounded solution by fairly distributing contributions (like Shapley values from cooperative game theory).

## 💡 How it works

This module:

1. **Generates synthetic financial data**, including features like `Credit_Score`, `Income`, `Debt`, `Years_Employed`, `Age`, and `Num_Accounts`.
2. **Trains a Random Forest classifier** on this data to predict credit approval decisions.
3. **Computes SHAP values** to attribute the prediction score to each feature for each sample.
4. **Visualizes feature impacts** with SHAP summary plots and force plots.

### Feature explanation logic

- Features with higher SHAP values push predictions towards approval (or rejection).
- Visualizations highlight both **global importance** (overall across dataset) and **local importance** (specific to each instance).

## 🧑‍💻 Example usage

```python
python financial_xai.py
