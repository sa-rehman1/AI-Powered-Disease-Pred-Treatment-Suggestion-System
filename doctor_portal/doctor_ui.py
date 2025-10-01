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

from dotenv import load_dotenv
load_dotenv()
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
    """Categorize risk level - defaults to Low if no prediction"""
    if pd.isna(prob):
        return "Low"  # Default to Low risk if no prediction available
    if prob >= 0.7:
        return "High"
    elif prob >= 0.3:
        return "Moderate"
    else:
        return "Low"

# ---------- Doctor UI ----------

def doctor_page():
    
    # --- Login / Signup ---
    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False

    if not st.session_state.doctor_logged_in:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Signup"])
        with tab1:
            st.markdown('<h3 style="color: #1b5e20; text-align: center;">Doctor Login</h3>', unsafe_allow_html=True)
            username = st.text_input("Username", key="doc_login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="doc_login_pw", placeholder="Enter your password")
            if st.button("Login", key="doc_login_btn"):
                success, msg = login_doctor(username, password)
                if success:
                    st.success(msg)
                    st.session_state.doctor_logged_in = True
                    st.session_state.doctor_username = username
                    st.rerun()
                else:
                    st.error(msg)
        with tab2:
            st.markdown('<h3 style="color: #1b5e20; text-align: center;">Create Doctor Account</h3>', unsafe_allow_html=True)
            username = st.text_input("New Username", key="doc_signup_user", placeholder="Choose a username")
            password = st.text_input("New Password", type="password", key="doc_signup_pw", placeholder="Choose a password")
            if st.button("Signup", key="doc_signup_btn"):
                success, msg = signup_doctor(username, password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        return

    # Welcome header with logout
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f'<h2 style="color: #1b5e20;">👨‍⚕️ Welcome, Dr. {st.session_state.doctor_username}</h2>', unsafe_allow_html=True)
    with col2:
        if st.button("Logout", key="doctor_logout"):
            st.session_state.doctor_logged_in = False
            st.session_state.doctor_username = None
            st.rerun()

    st.markdown("---")
    
    # Navigation menu
    menu_option = st.radio(
        "Navigation:",
        ["📊 All Patients Summary", "👤 Patient Details"],
        horizontal=True
    )
    
    st.markdown("---")

    # --- Load all reports ---
    files = get_all_patient_reports()
    if not files:
        st.info("No patient reports found yet.")
        return

    # OPTION 1: View All Patients Summary
    if menu_option == "📊 All Patients Summary":
        st.subheader("📊 All Patients Health Summary")
        
        summary_rows = []

        for f in files:
            patient_name = f.stem.replace("_data", "")
            df = pd.read_csv(f)
            if df.empty:
                continue
            latest_row = df.sort_values("timestamp", ascending=False).head(1).squeeze()

            # Get predictions with try-except
            try:
                diabetes_prob = predict_diabetes(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]) else np.nan
            except Exception:
                diabetes_prob = np.nan
                
            try:
                heart_prob = predict_heart(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]) else np.nan
            except Exception:
                heart_prob = np.nan

            # Categorize risks (defaults to Low if no prediction)
            diabetes_risk = categorize_risk(diabetes_prob)
            heart_risk = categorize_risk(heart_prob)
            
            # Overall risk is the highest of the two
            risk_priority = {"Low": 0, "Moderate": 1, "High": 2}
            overall_risk = diabetes_risk if risk_priority[diabetes_risk] >= risk_priority[heart_risk] else heart_risk
            
            summary_rows.append({
                "Patient": patient_name,
                "Diabetes Risk": diabetes_risk,
                "Heart Risk": heart_risk,
                "Overall Risk": overall_risk,
                "Last Report": df["timestamp"].max() if "timestamp" in df.columns else "Unknown"
            })

        summary_df = pd.DataFrame(summary_rows)

        # Display with color-coded risks
        def color_risk(val):
            if val=="High":
                return 'background-color: #ef5350; color: white; font-weight: bold'
            elif val=="Moderate":
                return 'background-color: #ffb74d; color: black; font-weight: bold'
            elif val=="Low":
                return 'background-color: #66bb6a; color: white; font-weight: bold'
            else:
                return ''

        st.dataframe(summary_df.style.applymap(color_risk, subset=["Overall Risk", "Diabetes Risk", "Heart Risk"]), use_container_width=True)

        # --- Statistics Overview ---
        st.markdown("### 📈 Statistics Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(summary_df)
        high_count = len(summary_df[summary_df["Overall Risk"]=="High"])
        mod_count = len(summary_df[summary_df["Overall Risk"]=="Moderate"])
        low_count = len(summary_df[summary_df["Overall Risk"]=="Low"])
        
        with col1:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(27, 94, 32, 0.1);">
                <h3 style="color: #1b5e20; margin: 0;">{total_patients}</h3>
                <p style="color: #388e3c; margin: 5px 0 0 0;">Total Patients</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(239, 83, 80, 0.2);">
                <h3 style="color: #c62828; margin: 0;">{high_count}</h3>
                <p style="color: #d32f2f; margin: 5px 0 0 0;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(255, 183, 77, 0.2);">
                <h3 style="color: #f57c00; margin: 0;">{mod_count}</h3>
                <p style="color: #ef6c00; margin: 5px 0 0 0;">Moderate Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: white; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 8px rgba(102, 187, 106, 0.2);">
                <h3 style="color: #2e7d32; margin: 0;">{low_count}</h3>
                <p style="color: #388e3c; margin: 5px 0 0 0;">Low Risk</p>
            </div>
            """, unsafe_allow_html=True)

        # --- High Priority Alerts ---
        st.markdown("### 🚨 High Priority Alerts")
        high_risk_df = summary_df[summary_df["Overall Risk"]=="High"]
        if high_risk_df.empty:
            st.success("✅ No high-risk patients currently")
        else:
            st.warning(f"⚠️ {len(high_risk_df)} patient(s) require immediate attention")
            st.dataframe(high_risk_df, use_container_width=True)

    # OPTION 2: View Patient-Wise Summary
    elif menu_option == "👤 Patient Details":
        st.subheader("👤 Patient-Wise Detailed Analysis")
        
        patient_names = [f.stem.replace("_data", "") for f in files]
        selected_patient = st.selectbox("Select Patient", ["Select a patient..."] + patient_names, index=0)

        if selected_patient != "Select a patient...":
            patient_file = Path(PATIENT_DATA_DIR) / f"{selected_patient}_data.csv"
            df = pd.read_csv(patient_file)
            if df.empty:
                st.info("No data available for this patient")
            else:
                # Patient info header
                st.markdown(f"""
                <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(27, 94, 32, 0.1);">
                    <h3 style="color: #1b5e20; margin: 0;">Patient: {selected_patient}</h3>
                    <p style="color: #388e3c; margin: 5px 0 0 0;">Detailed health analysis and history</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Last 6 reports (latest + previous 5)
                df_sorted = df.sort_values("timestamp", ascending=False)
                latest_report = df_sorted.head(1)
                prev_reports = df_sorted.iloc[1:6]

                st.markdown(f'### 📋 Current Situation ({latest_report["timestamp"].values[0]})')
                try:
                    insights = generate_patient_insights(df_sorted)
                    st.markdown(insights)
                except Exception as e:
                    st.info(f"Insights not available: {e}")

                st.markdown("### 📊 Previous 5 Reports")
                if not prev_reports.empty:
                    st.dataframe(prev_reports, use_container_width=True)
                else:
                    st.info("No previous reports available")

                # Trend analysis
                st.markdown("### 📈 Trend Analysis")
                numeric_cols = ["Glucose", "BMI", "Cholesterol", "BloodPressure", "MaxHR"]
                trend_data = []
                
                for col in numeric_cols:
                    if col in df_sorted.columns:
                        vals = df_sorted[col].head(6).values
                        if len(vals) >= 2:
                            try:
                                vals_numeric = [float(v) for v in vals if pd.notna(v)]
                                if len(vals_numeric) >= 2:
                                    change = vals_numeric[0] - vals_numeric[-1]
                                    if abs(change) < 1:
                                        trend = "Stable"
                                    elif change < 0:
                                        trend = "Worsening"
                                    else:
                                        trend = "Improving"
                                    
                                    trend_data.append({
                                        "Metric": col,
                                        "Current": f"{vals_numeric[0]:.1f}",
                                        "Previous": f"{vals_numeric[-1]:.1f}",
                                        "Change": f"{change:+.1f}",
                                        "Trend": trend
                                    })
                            except Exception:
                                continue
                
                if trend_data:
                    trend_df = pd.DataFrame(trend_data)
                    st.dataframe(trend_df, use_container_width=True)
                else:
                    st.info("No numeric trend data available")

                # Risk predictions
                st.markdown("### 🔬 Risk Predictions")
                latest_row = df_sorted.head(1).squeeze()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if all(col in df.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]):
                        try:
                            label, prob = predict_diabetes(latest_row.to_frame().T)
                            risk_cat = categorize_risk(prob)
                            if risk_cat == "High":
                                st.error(f"🔴 **Diabetes Risk: {risk_cat}** ({prob:.2f})")
                            elif risk_cat == "Moderate":
                                st.warning(f"🟡 **Diabetes Risk: {risk_cat}** ({prob:.2f})")
                            else:
                                st.success(f"🟢 **Diabetes Risk: {risk_cat}** ({prob:.2f})")
                        except Exception:
                            st.info("Diabetes prediction not available")
                    else:
                        st.success("🟢 **Diabetes Risk: Low** (Insufficient data)")
                
                with col2:
                    if all(col in df.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]):
                        try:
                            label, prob = predict_heart(latest_row.to_frame().T)
                            risk_cat = categorize_risk(prob)
                            if risk_cat == "High":
                                st.error(f"🔴 **Heart Disease Risk: {risk_cat}** ({prob:.2f})")
                            elif risk_cat == "Moderate":
                                st.warning(f"🟡 **Heart Disease Risk: {risk_cat}** ({prob:.2f})")
                            else:
                                st.success(f"🟢 **Heart Disease Risk: {risk_cat}** ({prob:.2f})")
                        except Exception:
                            st.info("Heart disease prediction not available")
                    else:
                        st.success("🟢 **Heart Disease Risk: Low** (Insufficient data)")

                # Follow-up reminders
                st.markdown("### ⏰ Follow-up Reminders")
                rem_df = get_reminders_for_patient(selected_patient)
                if rem_df.empty:
                    st.info("No reminders set for this patient")
                else:
                    st.dataframe(rem_df, use_container_width=True)

                # Download full patient report
                st.markdown("### 📥 Download Patient Report")
                with open(patient_file, "rb") as fh:
                    st.download_button(
                        f"📄 Download Report",
                        data=fh,
                        file_name=patient_file.name,
                        mime="text/csv",
                        use_container_width=True
                    )