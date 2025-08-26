from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load model + encoders
# -----------------------
ensemble = joblib.load("ensemble_model_compressed.pkl")
label_encoders = joblib.load("label_encoders_compressed.pkl")
target_le = joblib.load("target_encoder_compressed.pkl")

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Medical Device Recall Classifier")

# -----------------------
# Add CORS Middleware
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],   # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],   # Allows Content-Type and other headers
)

# -----------------------
# Input schema
# -----------------------
class InputData(BaseModel):
    classification: str
    code: str
    implanted: str
    name_device: str
    name_manufacturer: str

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(data: InputData):
    user_input = data.dict()

    # Fill missing features with "Unknown"
    expected_features = list(label_encoders.keys())
    input_dict = {col: "Unknown" for col in expected_features}
    input_dict.update(user_input)

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features and handle unknowns
    unknown_replacements = {}
    for col, le in label_encoders.items():
        val = input_df[col].iloc[0]
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")
            input_df[col] = ["Unknown"]
            unknown_replacements[col] = val
        input_df[col] = le.transform(input_df[col])

    # Predict probabilities
    pred_class_prob = ensemble.predict_proba(input_df)[0]

    # Take the class with the highest probability
    pred_class_idx = np.argmax(pred_class_prob)

    # Decode predicted class label
    pred_class_label = target_le.inverse_transform([pred_class_idx])[0]

    # Return JSON with rounded probabilities
    return {
        "predicted_class": pred_class_label,
        "class_probabilities": {
            k: round(float(v), 3) for k, v in zip(target_le.classes_, pred_class_prob)
        },
        "unknown_replacements": unknown_replacements
    }
