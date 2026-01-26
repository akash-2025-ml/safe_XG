from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pickle


import os
import warnings
import absl.logging
import pandas as pd

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress protobuf mismatch warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# Hide absl warnings
absl.logging.set_verbosity(absl.logging.ERROR)


# import shap
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "xg-safe-v2-23-1.pkll")

with open(
    model_path,
    "rb",
) as f:
    xg_model = pickle.load(f)

model_path = os.path.join(BASE_DIR, "label_encoder-output-safe-26-1.pkl")
with open(
    model_path,
    "rb",
) as f:
    en = pickle.load(f)

########################################################################
categorical_columns = ["dmarc_result", "dkim_result", "spf_result"]

################################################################################
feature_names = [
    "dmarc_result",
    "dkim_result",
    "spf_result",
    "sender_domain_reputation_score",
    "smtp_ip_reputation_score",
    "url_reputation_score",
    "content_spam_score",
    "malicious_attachment_count",
    "total_components_detected_malicious",
    "url_count",
]


##############################################################################################
default_values = {col: 0 for col in feature_names}  # default numeric value

default_values.update(
    {
        "dmarc_result": "pass",
        "dkim_result": "pass",
        "spf_result": "pass",
        "sender_domain_reputation_score": 0.95,
        "smtp_ip_reputation_score": 0.95,
        "url_reputation_score": 0.95,
        "content_spam_score": 0.01,
        "malicious_attachment_count": 0,
        "total_components_detected_malicious": 0,
        "url_count": 0,
    }
)

model_path = os.path.join(BASE_DIR, "preprocessor-safe-26.pkl")
preprocessor = joblib.load(model_path)

app = FastAPI()
l = []


# Input schema
class InputData(BaseModel):
    data: dict  # {"feature_name": value, ...}


@app.get("/")
def home():
    return {"message": "Welcome to the ML API! Send POST to /predict with JSON data."}


@app.post("/predict")
def predict(request: InputData):
    """
    Correct prediction function using OneHotEncoder + ColumnTransformer
    """
    data_dict = request.data
    # Extract data
    # data_dict = input_data["data"]

    # Create DataFrame
    df = pd.DataFrame([data_dict])

    # Ensure all expected columns exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = default_values.get(col, "none")

    df = df.replace({True: 1, False: 0})

    X_encoded = preprocessor.transform(df)

    # Predict
    # return str(X_encoded)
    probs = xg_model.predict(X_encoded)
    print(probs)
    pred_class = np.argmax(probs, axis=1)
    print(pred_class)
    # Decode output label
    prediction_label = en.inverse_transform(pred_class)[0]

    # # return prediction_label
    return {
        "predicted_class": prediction_label,
        "predicted_class by ANN": str(max(probs[0])),
    }
