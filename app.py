import gradio as gr
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load("credit_card_model")

def predict_fraud(csv_file):
    df = pd.read_csv(csv_file.name)

    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)
    if 'Class' in df.columns:
        df = df.drop(['Class'], axis=1)

    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    preds = model.predict(df)
    df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in preds]

    summary = df['Prediction'].value_counts().to_frame(name='Count').reset_index().rename(columns={'index': 'Class'})
    summary_str = summary.to_string(index=False)

    output_path = "predictions_output.csv"
    df.to_csv(output_path, index=False)

    return df, summary_str, output_path

iface = gr.Interface(
    fn=predict_fraud,
    inputs=gr.File(label="Upload Credit Card CSV"),
    outputs=[
        gr.Dataframe(label="Predictions"),
        gr.Textbox(label="Class Distribution Summary"),
        gr.File(label="Download Predictions CSV")
    ],
    title="🔍 Credit Card Fraud Detection",
    description="Upload a CSV file with transactions. The model will predict if each is Fraud or Not Fraud."
)

if __name__ == "__main__":
    iface.launch()
