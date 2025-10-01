# doctor_ui.py
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from hashlib import sha256
import io

from models.ml_pipeline import predict_diabetes, predict_heart
from patient_portal.patient_ui import hash_password, PATIENT_DATA_DIR, REMINDERS_FILE, generate_patient_insights

# ---------- Helper functions ----------

DOCTOR_FILE = "data/doctors.csv"
os.makedirs("data", exist_ok=True)

def save_doctors(df):
    df.to_csv(DOCTOR_FILE, index=False)

def load_doctors():
    if os.path.exists(DOCTOR_FILE):
        return pd.read_csv(DOCTOR_FILE)
    return pd.DataFrame(columns=["username", "password"])

def signup_doctor(username, password):
    doctors = load_doctors()
    if username in doctors["username"].values:
        return False, "Doctor username already exists!"
    new_doc = pd.DataFrame([{"username": username, "password": hash_password(password)}])
    doctors = pd.concat([doctors, new_doc], ignore_index=True)
    save_doctors(doctors)
    return True, "Signup successful! You can now login."

def login_doctor(username, password):
    doctors = load_doctors()
    if username not in doctors["username"].values:
        return False, "Username does not exist!"
    user_pw = doctors.loc[doctors["username"]==username, "password"].values[0]
    if hash_password(password) == user_pw:
        return True, "Login successful!"
    return False, "Incorrect password!"

def get_all_patient_reports():
    """Return list of all patient CSV files"""
    p = Path(PATIENT_DATA_DIR)
    if not p.exists():
        return []
    files = sorted(p.glob("*_data.csv"), key=lambda x: x.stat().st_mtime, reverse=True)
    return files

def get_reminders_for_patient(patient_name):
    if os.path.exists(REMINDERS_FILE):
        rem_df = pd.read_csv(REMINDERS_FILE)
        return rem_df[rem_df["username"]==patient_name].sort_values("reminder_date")
    return pd.DataFrame()

def categorize_risk(prob):
    if pd.isna(prob):
        return "Unknown"
    if prob >= 0.7:
        return "High"
    elif prob >= 0.3:
        return "Moderate"
    else:
        return "Low"

# ---------- Doctor UI ----------

def doctor_page():
    st.title("Doctor Portal")

    # --- Login / Signup ---
    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False

    if not st.session_state.doctor_logged_in:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        with tab1:
            st.subheader("Doctor Login")
            username = st.text_input("Username", key="doc_login_user")
            password = st.text_input("Password", type="password", key="doc_login_pw")
            if st.button("Login", key="doc_login_btn"):
                success, msg = login_doctor(username, password)
                st.info(msg)
                if success:
                    st.session_state.doctor_logged_in = True
                    st.session_state.doctor_username = username
        with tab2:
            st.subheader("Doctor Signup")
            username = st.text_input("New Username", key="doc_signup_user")
            password = st.text_input("New Password", type="password", key="doc_signup_pw")
            if st.button("Signup", key="doc_signup_btn"):
                success, msg = signup_doctor(username, password)
                st.info(msg)
        return

    st.subheader(f"Welcome, Dr. {st.session_state.doctor_username}")

    # --- Load all reports ---
    files = get_all_patient_reports()
    if not files:
        st.info("No patient reports found yet.")
        return

    # --- AI Health Summary ---
    st.subheader("Health Summary for All Patients")
    summary_text = []
    summary_rows = []

    for f in files:
        patient_name = f.stem.replace("_data", "")
        df = pd.read_csv(f)
        if df.empty:
            continue
        latest_row = df.sort_values("timestamp", ascending=False).head(1).squeeze()

        diabetes_prob = predict_diabetes(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]) else np.nan
        heart_prob = predict_heart(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]) else np.nan

        overall_risk = max(categorize_risk(diabetes_prob), categorize_risk(heart_prob), key=lambda x: ["Low","Moderate","High","Unknown"].index(x))
        summary_rows.append({
            "Patient": patient_name,
            "Diabetes Risk": diabetes_prob,
            "Heart Risk": heart_prob,
            "Overall Risk": overall_risk,
            "Last Report": df["timestamp"].max() if "timestamp" in df.columns else "Unknown"
        })

    summary_df = pd.DataFrame(summary_rows)

    # Display with color-coded risks
    def color_risk(val):
        if val=="High":
            color = 'background-color: red; color: white'
        elif val=="Moderate":
            color = 'background-color: yellow; color: black'
        elif val=="Low":
            color = 'background-color: green; color: white'
        else:
            color = ''
        return color

    st.dataframe(summary_df.style.applymap(color_risk, subset=["Overall Risk"]))

    # --- High Priority Alerts ---
    st.subheader("High Priority Alerts")
    high_risk_df = summary_df[summary_df["Overall Risk"]=="High"]
    if high_risk_df.empty:
        st.info("No high-risk patients currently.")
    else:
        st.dataframe(high_risk_df)

    # --- Detailed Patient Analysis ---
    st.subheader("Patient Detailed Analysis")
    patient_names = [f.stem.replace("_data", "") for f in files]
    selected_patient = st.selectbox("Select patient", ["No Patient Selected"] + patient_names, index=0)

    if selected_patient != "No Patient Selected":
        patient_file = Path(PATIENT_DATA_DIR) / f"{selected_patient}_data.csv"
        df = pd.read_csv(patient_file)
        if df.empty:
            st.info("No data for this patient")
        else:
            # Last 6 reports (latest + previous 5)
            df_sorted = df.sort_values("timestamp", ascending=False)
            latest_report = df_sorted.head(1)
            prev_reports = df_sorted.iloc[1:6]

            st.markdown(f"### Current Situation ({latest_report['timestamp'].values[0]})")
            try:
                insights = generate_patient_insights(df_sorted)
                st.markdown(insights)
            except Exception as e:
                st.info(f"Insights not available: {e}")

            st.markdown("### Previous 5 Reports")
            st.dataframe(prev_reports)

            # Trend analysis (better/worse)
            st.markdown("### Trend Analysis")
            numeric_cols = ["Glucose", "BMI", "Cholesterol", "BloodPressure", "MaxHR"]
            trend_lines = []
            for col in numeric_cols:
                if col in df_sorted.columns:
                    vals = df_sorted[col].head(6).values  # latest 6
                    if len(vals) >= 2:
                        trend = "improving" if vals[0] < vals[-1] else "worsening" if vals[0] > vals[-1] else "stable"
                        trend_lines.append(f"- {col}: {trend}")
            if trend_lines:
                st.markdown("\n".join(trend_lines))
            else:
                st.markdown("No numeric trend data available.")

            # Follow-up reminders
            st.subheader("Follow-up Reminders")
            rem_df = get_reminders_for_patient(selected_patient)
            if rem_df.empty:
                st.info("No reminders set for this patient.")
            else:
                st.dataframe(rem_df)

            # Download full patient report
            with open(patient_file, "rb") as fh:
                st.download_button(f"Download full report for {selected_patient}", data=fh, file_name=patient_file.name)

