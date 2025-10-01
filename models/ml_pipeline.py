# ml_pipeline.py
# models/ml_pipeline.py

import pandas as pd
import pickle
import os

# Paths to saved models
BASE_DIR = os.path.dirname(__file__)
DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "heart_model.pkl")

# Load the trained models
with open(DIABETES_MODEL_PATH, "rb") as f:
    diabetes_model = pickle.load(f)

with open(HEART_MODEL_PATH, "rb") as f:
    heart_model = pickle.load(f)


# -----------------------
# Diabetes Prediction
# -----------------------
def predict_diabetes(df: pd.DataFrame):
    """
    Predict Diabetes risk from patient data (DataFrame).
    Returns: risk_label (str), risk_prob (float 0-1)
    """
    required_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                     "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return "Cannot predict (missing columns)", None

    X = df[required_cols]

    # Predict probability of Outcome=1 (Diabetes)
    prob = diabetes_model.predict_proba(X)[:,1].mean()  # mean if multiple rows
    risk_label = "High Risk of Diabetes" if prob >= 0.5 else "Low Risk of Diabetes"
    return risk_label, prob


# -----------------------
# Heart Disease Prediction
# -----------------------
def predict_heart(df: pd.DataFrame):
    """
    Predict Heart Disease risk from patient data (DataFrame).
    Returns: risk_label (str), risk_prob (float 0-1)
    """
    required_cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol",
                     "FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return "Cannot predict (missing columns)", None

    X = df[required_cols].copy()

    # Encode categorical columns for prediction
    cat_cols = ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"]
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Ensure all model columns exist (add missing dummy columns as 0)
    model_cols = heart_model.feature_names_in_
    for col in model_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[model_cols]

    # Predict probability
    prob = heart_model.predict_proba(X)[:,1].mean()
    risk_label = "High Risk of Heart Disease" if prob >= 0.5 else "Low Risk of Heart Disease"
    return risk_label, prob
