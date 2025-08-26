import os
import warnings
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.exceptions import InconsistentVersionWarning

# -----------------------
# Suppress LabelEncoder version warnings
# -----------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="Medical Device Recall Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Root endpoint (health check)
# -----------------------
@app.get("/")
def root():
    return {"message": "Medical Device Recall Classifier API is running!"}

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
# Lazy-loading with caching
# -----------------------
ensemble = None
label_encoders = None
target_le = None

def load_models():
    global ensemble, label_encoders, target_le
    if ensemble is None or label_encoders is None or target_le is None:
        ensemble = joblib.load("ensemble_model_compressed.pkl")
        label_encoders = joblib.load("label_encoders_compressed.pkl")
        target_le = joblib.load("target_encoder_compressed.pkl")
    return ensemble, label_encoders, target_le

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict(data: InputData):
    # Load models lazily
    ensemble_model, encoders, target_encoder = load_models()

    user_input = data.dict()

    # Fill missing features with "Unknown"
    expected_features = list(encoders.keys())
    input_dict = {col: "Unknown" for col in expected_features}
    input_dict.update(user_input)

    input_df = pd.DataFrame([input_dict])

    # Encode categorical features and handle unknowns
    unknown_replacements = {}
    for col, le in encoders.items():
        val = input_df[col].iloc[0]
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, "Unknown")
            input_df[col] = ["Unknown"]
            unknown_replacements[col] = val
        input_df[col] = le.transform(input_df[col])

    # Predict probabilities
    pred_class_prob = ensemble_model.predict_proba(input_df)[0]

    # Take the class with the highest probability
    pred_class_idx = np.argmax(pred_class_prob)

    # Decode predicted class label
    pred_class_label = target_encoder.inverse_transform([pred_class_idx])[0]

    # Return JSON with rounded probabilities
    return {
        "predicted_class": pred_class_label,
        "class_probabilities": {
            k: round(float(v), 3) for k, v in zip(target_encoder.classes_, pred_class_prob)
        },
        "unknown_replacements": unknown_replacements
    }

# -----------------------
# Entry point for uvicorn / Render
# -----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Dynamic port for Render
    uvicorn.run(app, host="0.0.0.0", port=port)
