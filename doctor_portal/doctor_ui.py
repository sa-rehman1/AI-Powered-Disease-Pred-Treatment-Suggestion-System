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


st.markdown("""
<style>
    .doctor-header {
        background: linear-gradient(135deg, #1e40af 0%, #3730a3 100%);
        color: white;
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .stats-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border-top: 5px solid;
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
    }
    
    .patient-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    .risk-high { 
        border-left-color: #dc3545; 
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
    }
    
    .risk-moderate { 
        border-left-color: #ffc107; 
        background: linear-gradient(135deg, #ffffff 0%, #fffbf0 100%);
    }
    
    .risk-low { 
        border-left-color: #28a745; 
        background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
    }
    
    .alert-banner {
        background: linear-gradient(135deg, #dc3545, #c53030);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .login-container {
        background: white;
        border-radius: 20px;
        padding: 45px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid #e9ecef;
        margin: 0 auto;
        max-width: 600px;
    }
    
    .back-button {
        background: #6c757d !important;
        color: white !important;
        border: none !important;
    }
    
    /* Improve radio button visibility */
    .stRadio > div {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stRadio > div > label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #2d3748 !important;
    }
    
    .insight-box {
        background: white;
        color: #2d3748;
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .insight-box h1, .insight-box h2, .insight-box h3, .insight-box h4 {
        color: #1e40af;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 8px;
    }
    
    .insight-box h3:first-child, .insight-box h4:first-child {
        margin-top: 0;
    }
    
    /* Center the login page */
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
    }
</style>
""", unsafe_allow_html=True)

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
        return "Low"  
    if prob >= 0.7:
        return "High"
    elif prob >= 0.3:
        return "Moderate"
    else:
        return "Low"

# ---------- Doctor UI ----------

def doctor_page():
    
  
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("← Back to Main", key="back_to_role_doctor", use_container_width=True):
            st.session_state.current_page = "role_selection"
            st.session_state.doctor_logged_in = False
            st.session_state.doctor_username = None
            st.rerun()
    
    # --- Login / Signup ---
    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False

    if not st.session_state.doctor_logged_in:
        
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="login-container">
            <h2 style="color: #1e40af; margin-bottom: 10px; text-align: center;">👨‍⚕️ Doctor Portal</h2>
            <p style="color: #718096; text-align: center; margin-bottom: 30px;">Monitor patients and provide expert care</p>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 **Login**", "📝 **Signup**"])
        
        with tab1:
            st.markdown("### Doctor Login")
            st.markdown("Access the patient management dashboard")
            
            username = st.text_input("👤 Username", placeholder="Enter your username", key="doc_login_user")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="doc_login_pw")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🚀 Access Portal", key="doc_login_btn", use_container_width=True):
                    if username and password:
                        success, msg = login_doctor(username, password)
                        if success:
                            st.success(msg)
                            st.session_state.doctor_logged_in = True
                            st.session_state.doctor_username = username
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please enter both username and password")
            
            with col2:
                if st.button("🔄 Clear", key="clear_doc_login", use_container_width=True):
                    st.rerun()
        
        with tab2:
            st.markdown("### Create Doctor Account")
            st.markdown("Register for healthcare professional access")
            
            username = st.text_input("👤 Choose Username", placeholder="Pick a username", key="doc_signup_user")
            password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="doc_signup_pw")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🌟 Register Account", key="doc_signup_btn", use_container_width=True):
                    if username and password:
                        success, msg = signup_doctor(username, password)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error("Please enter both username and password")
            
            with col2:
                if st.button("🔄 Clear", key="clear_doc_signup", use_container_width=True):
                    st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

    # Doctor is logged in - show dashboard
    
    st.markdown(f"""
    <div class="doctor-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #1b5e20; font-size: 2.2rem;">👨‍⚕️ Welcome, Dr. {st.session_state.doctor_username}</h1>
                <p style="margin: 0; color: #2e7d32; font-size: 1.2rem;">Patient Management Dashboard</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation and logout
    col1, col2, col3 = st.columns([4, 1, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_doctor", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🚪 Logout", key="doctor_logout", use_container_width=True):
            st.session_state.current_page = "role_selection"
            st.session_state.doctor_logged_in = False
            st.session_state.doctor_username = None
            st.rerun()

    st.markdown("---")
    
    # Navigation menu with clear radio buttons
    st.markdown("### 📍 Navigation")
    menu_option = st.radio(
        "Choose a section:",
        ["📊 All Patients Summary", "👤 Patient Details"],
        horizontal=True,
        key="doctor_navigation"
    )
    
    st.markdown("---")

    # --- Load all reports ---
    files = get_all_patient_reports()
    if not files:
        st.info("👥 No patient reports found yet. Patients need to upload their medical reports first.")
        return

    # OPTION 1: View All Patients Summary
    if menu_option == "📊 All Patients Summary":
        st.markdown("## 📊 All Patients Health Summary")
        st.markdown("Overview of all patients and their current health risk status.")
        
        summary_rows = []

        for f in files:
            patient_name = f.stem.replace("_data", "")
            df = pd.read_csv(f)
            if df.empty:
                continue
            latest_row = df.sort_values("timestamp", ascending=False).head(1).squeeze()

            # Get predictions 
            try:
                diabetes_prob = predict_diabetes(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]) else np.nan
            except Exception:
                diabetes_prob = np.nan
                
            try:
                heart_prob = predict_heart(latest_row.to_frame().T)[1] if all(col in df.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]) else np.nan
            except Exception:
                heart_prob = np.nan

            # Categorize risks 
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
                return 'background-color: #fecaca; color: #dc2626; font-weight: bold; padding: 8px; border-radius: 6px;'
            elif val=="Moderate":
                return 'background-color: #fef3c7; color: #d97706; font-weight: bold; padding: 8px; border-radius: 6px;'
            elif val=="Low":
                return 'background-color: #d1fae5; color: #059669; font-weight: bold; padding: 8px; border-radius: 6px;'
            else:
                return 'padding: 8px;'

        st.markdown("### 📋 Patient Risk Overview")
        st.dataframe(
            summary_df.style.applymap(color_risk, subset=["Overall Risk", "Diabetes Risk", "Heart Risk"]), 
            use_container_width=True
        )

        # --- Statistics Overview ---
        st.markdown("### 📈 Practice Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_patients = len(summary_df)
        high_count = len(summary_df[summary_df["Overall Risk"]=="High"])
        mod_count = len(summary_df[summary_df["Overall Risk"]=="Moderate"])
        low_count = len(summary_df[summary_df["Overall Risk"]=="Low"])
        
        with col1:
            st.markdown(f"""
            <div class="stats-card" style="border-top-color: #1e40af;">
                <h3 style="color: #1e40af; margin: 0; font-size: 2.2rem;">{total_patients}</h3>
                <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 500;">Total Patients</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card" style="border-top-color: #dc2626;">
                <h3 style="color: #dc2626; margin: 0; font-size: 2.2rem;">{high_count}</h3>
                <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 500;">High Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stats-card" style="border-top-color: #d97706;">
                <h3 style="color: #d97706; margin: 0; font-size: 2.2rem;">{mod_count}</h3>
                <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 500;">Moderate Risk</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stats-card" style="border-top-color: #059669;">
                <h3 style="color: #059669; margin: 0; font-size: 2.2rem;">{low_count}</h3>
                <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem; font-weight: 500;">Low Risk</p>
            </div>
            """, unsafe_allow_html=True)

       
        # --- High Priority Alerts ---
        st.markdown("### 🚨 High Priority Alerts")
        high_risk_df = summary_df[summary_df["Overall Risk"]=="High"]
        if high_risk_df.empty:
            st.success("✅ No high-risk patients currently. All patients are stable.")
        else:
            st.markdown(f"""
            <div class="alert-banner">
                <h3 style="margin: 0; color: #8b0000; font-weight: 700;">⚠️ {len(high_risk_df)} Patient(s) Require Immediate Attention</h3>
                <p style="margin: 10px 0 0 0; color: #2d3748; font-weight: 600; font-size: 1.1rem;">Please review these patients as soon as possible</p>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(high_risk_df, use_container_width=True)

    # OPTION 2: View Patient-Wise Summary
    elif menu_option == "👤 Patient Details":
        st.markdown("## 👤 Patient-Wise Detailed Analysis")
        st.markdown("Select a patient to view detailed health analysis and medical history.")
        
        patient_names = [f.stem.replace("_data", "") for f in files]
        selected_patient = st.selectbox("Select Patient", ["Select a patient..."] + patient_names, index=0)

        if selected_patient != "Select a patient...":
            patient_file = Path(PATIENT_DATA_DIR) / f"{selected_patient}_data.csv"
            df = pd.read_csv(patient_file)
            if df.empty:
                st.info("📝 No data available for this patient yet.")
            else:
                # Patient info header
                st.markdown(f"""
                <div class="patient-card">
                    <h2 style="color: #1e40af; margin: 0;">Patient: {selected_patient}</h2>
                    <p style="color: #6b7280; margin: 10px 0 0 0; font-size: 1.1rem;">
                        Detailed health analysis and medical history
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Last 6 reports (latest + previous 5)
                df_sorted = df.sort_values("timestamp", ascending=False)
                latest_report = df_sorted.head(1)
                prev_reports = df_sorted.iloc[1:6]

                st.markdown(f'### 📋 Current Health Situation')
                st.markdown(f"*Latest report from: {latest_report['timestamp'].values[0]}*")

                with st.spinner("🔄 Generating AI insights..."):
                    try:
                        insights = generate_patient_insights(df_sorted)
                        
                        st.markdown(insights)
                    except Exception as e:
                        st.info(f"ℹ️ AI insights temporarily unavailable: {e}")

                st.markdown("### 📊 Previous Medical Reports")
                if not prev_reports.empty:
                    st.dataframe(prev_reports, use_container_width=True)
                else:
                    st.info("📝 No previous reports available for this patient.")

             

                # Trend analysis
                st.markdown("### 📈 Health Metrics Trend Analysis")
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
                                        trend = "🟰 Stable"
                                    elif change < 0:
                                        trend = "📉 Improving"
                                    else:
                                        trend = "📈 Worsening"
                                    
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
                    st.info("📊 Not enough numeric data available for trend analysis.")

                # Risk predictions
                st.markdown("### 🔬 Disease Risk Predictions")
                latest_row = df_sorted.head(1).squeeze()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if all(col in df.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]):
                        try:
                            label, prob = predict_diabetes(latest_row.to_frame().T)
                            risk_cat = categorize_risk(prob)
                            if risk_cat == "High":
                                st.error(f"🔴 **Diabetes Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                            elif risk_cat == "Moderate":
                                st.warning(f"🟡 **Diabetes Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                            else:
                                st.success(f"🟢 **Diabetes Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                        except Exception:
                            st.info("ℹ️ Diabetes prediction not available")
                    else:
                        st.success("🟢 **Diabetes Risk: Low**  \n(Insufficient data for prediction)")
                
                with col2:
                    if all(col in df.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]):
                        try:
                            label, prob = predict_heart(latest_row.to_frame().T)
                            risk_cat = categorize_risk(prob)
                            if risk_cat == "High":
                                st.error(f"🔴 **Heart Disease Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                            elif risk_cat == "Moderate":
                                st.warning(f"🟡 **Heart Disease Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                            else:
                                st.success(f"🟢 **Heart Disease Risk: {risk_cat}**  \nProbability: {prob:.2f}")
                        except Exception:
                            st.info("ℹ️ Heart disease prediction not available")
                    else:
                        st.success("🟢 **Heart Disease Risk: Low**  \n(Insufficient data for prediction)")

                # Follow-up reminders
                st.markdown("### ⏰ Patient Reminders")
                rem_df = get_reminders_for_patient(selected_patient)
                if rem_df.empty:
                    st.info("📝 No reminders set for this patient.")
                else:
                    st.dataframe(rem_df, use_container_width=True)

                # Download full patient report
                st.markdown("### 📥 Download Patient Report")
                st.markdown("Download the complete patient data for offline review or records.")
                
                with open(patient_file, "rb") as fh:
                    st.download_button(
                        f"📄 Download {selected_patient}'s Complete Report",
                        data=fh,
                        file_name=patient_file.name,
                        mime="text/csv",
                        use_container_width=True,
                        help="Download the complete CSV file containing all patient data"
                    )