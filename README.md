# 💳 Credit Card Fraud Detection App

This is a machine learning application that detects fraudulent credit card transactions using supervised learning models like **Logistic Regression**, **Decision Tree**, and **Random Forest**. It includes a simple **Gradio-based web interface** for making predictions from uploaded CSV files.

---

## 🚀 Live Features

- ✅ Upload CSV with transactions
- ✅ Preprocess and normalize features
- ✅ Predict fraud vs. non-fraud using trained model
- ✅ View prediction summary and download results
- ✅ CLI interface for batch processing

🧠 Model & Data

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features anonymized (V1–V28) + `Amount`
- Imbalanced dataset handled via:
  - ✅ Undersampling
  - ✅ SMOTE oversampling
- Best model: **Random Forest Classifier**
- Saved using `joblib`

---

## 🖥️ CLI Usage

```bash
python predict.py path/to/your_file.csv


@WithZahir