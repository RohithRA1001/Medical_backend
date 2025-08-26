import warnings
from sklearn.exceptions import InconsistentVersionWarning
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# Suppress LabelEncoder version warnings
# -----------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
# Version-proof predict function
# -----------------------
@app.post("/predict")
def predict(data: InputData):
    user_input = data.dict()

    # Prepare input dataframe
    expected_features = list(label_encoders.keys())
    input_dict = {col: "Unknown" for col in expected_features}
    input_dict.update(user_input)
    input_df = pd.DataFrame([input_dict])

    unknown_replacements = {}

    # Encode categorical features safely
    for col, le in label_encoders.items():
        val = input_df[col].iloc[0]

        # Use version-proof mapping instead of relying on classes_
        le_dict = {cls: idx for idx, cls in enumerate(le.classes_)}
        if val not in le_dict:
            val_encoded = le_dict.get("Unknown", len(le_dict))
            unknown_replacements[col] = val
        else:
            val_encoded = le_dict[val]

        input_df[col] = [val_encoded]

    # Predict probabilities
    pred_class_prob = ensemble.predict_proba(input_df)[0]

    # Get predicted class index
    pred_class_idx = int(np.argmax(pred_class_prob))

    # Decode class safely using a mapping
    target_dict = {idx: cls for idx, cls in enumerate(target_le.classes_)}
    pred_class_label = target_dict.get(pred_class_idx, "Unknown")

    # Return JSON with rounded probabilities
    class_probs = {
        cls: round(float(prob), 3)
        for cls, prob in zip(target_le.classes_, pred_class_prob)
    }

    return {
        "predicted_class": pred_class_label,
        "class_probabilities": class_probs,
        "unknown_replacements": unknown_replacements
    }
