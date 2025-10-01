# models/train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os

# -----------------------
# Diabetes Model
# -----------------------
print("Training Diabetes Model...")

diabetes_path = r"C:\Users\conta\OneDrive\Desktop\Projects\AI-Powered-Disease-Pred-Treatment-Suggestion-System\datsets\diabetes.csv"  # replace with your path

df = pd.read_csv(diabetes_path)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv_scores = cross_val_score(diabetes_model, X_train, y_train, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")

# Fit model
diabetes_model.fit(X_train, y_train)

# Test evaluation
y_pred = diabetes_model.predict(X_test)
print("Classification Report (Diabetes):")
print(classification_report(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
diabetes_model_path = os.path.join("models", "diabetes_model.pkl")
with open(diabetes_model_path, "wb") as f:
    pickle.dump(diabetes_model, f)
print(f"Diabetes model saved at {diabetes_model_path}")


# -----------------------
# Heart Disease Model
# -----------------------
print("\nTraining Heart Disease Model...")

heart_path = r"C:\Users\conta\OneDrive\Desktop\Projects\AI-Powered-Disease-Pred-Treatment-Suggestion-System\datsets\heart.csv"  # replace with your path
df = pd.read_csv(heart_path)

# Target column: HeartDisease
target_col = "HeartDisease"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
heart_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation
cv_scores = cross_val_score(heart_model, X_train, y_train, cv=5)
print(f"Cross-validation Accuracy: {cv_scores.mean():.4f}")

# Fit model
heart_model.fit(X_train, y_train)

# Test evaluation
y_pred = heart_model.predict(X_test)
print("Classification Report (Heart Disease):")
print(classification_report(y_test, y_pred))

# Save model
heart_model_path = os.path.join("models", "heart_model.pkl")
with open(heart_model_path, "wb") as f:
    pickle.dump(heart_model, f)
print(f"Heart Disease model saved at {heart_model_path}")
