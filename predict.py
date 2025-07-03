import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load("credit_card_model")

# Load CSV
if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/your_file.csv")
    sys.exit()

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

if 'Time' in df.columns:
    df = df.drop(['Time'], axis=1)
if 'Class' in df.columns:
    df = df.drop(['Class'], axis=1)

if 'Amount' in df.columns:
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

predictions = model.predict(df)
df['Prediction'] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]

print(df[['Prediction']].value_counts())
print("\nFirst few predictions:")
print(df[['Prediction']].head())

df.to_csv("predictions_output.csv", index=False)
print("\n✅ Results saved to predictions_output.csv")
