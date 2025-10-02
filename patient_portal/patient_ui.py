import streamlit as st
import pandas as pd
import os
from datetime import datetime
from hashlib import sha256
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io

from models.ml_pipeline import predict_diabetes, predict_heart
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Directories
USER_FILE = "data/users.csv"
PATIENT_DATA_DIR = "data/patients"
REMINDERS_FILE = "data/reminders.csv"
os.makedirs(PATIENT_DATA_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


st.markdown("""
<style>
    .patient-header {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border-left: 6px solid #2E8B57;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease;
        border: 1px solid #e9ecef;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .risk-low {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
    }
    
    .risk-moderate {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #ffffff 0%, #fffbf0 100%);
    }
    
    .risk-high {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
    }
    
    .upload-area {
        border: 3px dashed #2E8B57;
        border-radius: 20px;
        padding: 50px;
        text-align: center;
        background: #f8fff9;
        margin: 25px 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: #f0fff4;
        border-color: #267349;
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
        color: #2E8B57;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 8px;
    }
    
    .insight-box h3:first-child, .insight-box h4:first-child {
        margin-top: 0;
    }
    
    .reminder-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        border-left: 4px solid #2E8B57;
        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.06);
        border: 1px solid #e9ecef;
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
    
    .back-button:hover {
        background: #5a6268 !important;
        transform: translateY(-1px) !important;
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
    
    /* Center the login page */
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------

def hash_password(password):
    return sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_FILE):
        return pd.read_csv(USER_FILE)
    return pd.DataFrame(columns=["username", "password"])

def save_users(df):
    df.to_csv(USER_FILE, index=False)

def signup(username, password):
    users = load_users()
    if username in users["username"].values:
        return False, "Username already exists!"
    new_user = pd.DataFrame([{"username": username, "password": hash_password(password)}])
    users = pd.concat([users, new_user], ignore_index=True)
    save_users(users)
    return True, "Signup successful! You can now login."

def login(username, password):
    users = load_users()
    if username not in users["username"].values:
        return False, "Username does not exist!"
    user_pw = users.loc[users["username"]==username, "password"].values[0]
    if hash_password(password) == user_pw:
        return True, "Login successful!"
    return False, "Incorrect password!"

def save_patient_report(username, df):
    """Append uploaded report to patient CSV"""
    patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(patient_file):
        existing_df = pd.read_csv(patient_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv(patient_file, index=False)
    return patient_file

def get_last_reports(username, n=5):
    """Return last n reports as DataFrame"""
    patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
    if os.path.exists(patient_file):
        df = pd.read_csv(patient_file)
        if "timestamp" in df.columns:
            try:
                df["timestamp_parsed"] = pd.to_datetime(df["timestamp"])
                df = df.sort_values("timestamp_parsed", ascending=False)
                df = df.drop(columns=["timestamp_parsed"])
            except Exception:
                df = df.sort_values("timestamp", ascending=False)
        else:
            df = df.sort_values(df.columns[0], ascending=False)
        return df.head(n)
    return pd.DataFrame()

def generate_patient_insights(patient_df):
    """Generate structured insights using Groq LLM"""
    if patient_df.empty:
        return "No data available for insights."

    current_df = patient_df.sort_values("timestamp", ascending=False).head(1)
    previous_df = patient_df.sort_values("timestamp", ascending=False).iloc[1:6]

    def df_to_text(df):
        return "\n".join(
            df.apply(lambda r: ", ".join(f"{c}: {r[c]}" for c in df.columns if c != "timestamp"), axis=1)
        )

    current_text = df_to_text(current_df)
    previous_text = df_to_text(previous_df) if not previous_df.empty else "No previous reports."

    prompt_template = f"""
You are a medical AI assistant. Patient's medical history and current report:

Previous Reports:
{previous_text}

Current Report:
{current_text}

Please respond in the following structured format:

### Current Situation
Explain current health status and risk analysis

### Lifestyle Recommendations
Give actionable lifestyle advice

### Disease Education
Brief educational info about the disease

### Next Steps
What the patient should do next
"""

    model = ChatGroq(model="gemma2-9b-it")
    prompt = ChatPromptTemplate.from_messages([("human", prompt_template)])
    msg = model.predict_messages(prompt.format_messages())
    if isinstance(msg, list):
        return msg[0].content
    return msg.content

def create_health_summary_bytes(username, latest_row, latest_preds, insights_text):
    """Create a PDF summary bytes object if reportlab is available"""
    summary_text_lines = []
    summary_text_lines.append(f"Patient: {username}")
    summary_text_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_text_lines.append("")
    summary_text_lines.append("=== Latest Report Values ===")
    for c, v in latest_row.items():
        summary_text_lines.append(f"{c}: {v}")
    summary_text_lines.append("")
    summary_text_lines.append("=== Model Predictions ===")
    for k, v in latest_preds.items():
        if v is None or (isinstance(v, float) and np.isnan(v)):
            summary_text_lines.append(f"{k}: No prediction available")
        else:
            summary_text_lines.append(f"{k}: {v:.3f}")
    summary_text_lines.append("")
    summary_text_lines.append("=== AI Insights ===")
    summary_text_lines.append(insights_text)
    summary_text = "\n".join(summary_text_lines)

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch

        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        textobject = c.beginText()
        textobject.setTextOrigin(inch, height - inch)
        textobject.setFont("Helvetica", 10)

        for line in summary_text.split("\n"):
            textobject.textLine(line)
        c.drawText(textobject)
        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer.read(), "application/pdf", f"{username}_health_summary.pdf"
    except Exception:
        return summary_text.encode("utf-8"), "text/plain", f"{username}_health_summary.txt"

def save_reminder(username, reminder_date_str, note=""):
    """Save a reminder row to REMINDERS_FILE"""
    os.makedirs(os.path.dirname(REMINDERS_FILE), exist_ok=True)
    reminder_row = {
        "username": username,
        "reminder_date": reminder_date_str,
        "note": note,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    if os.path.exists(REMINDERS_FILE):
        rem_df = pd.read_csv(REMINDERS_FILE)
        rem_df = pd.concat([rem_df, pd.DataFrame([reminder_row])], ignore_index=True)
    else:
        rem_df = pd.DataFrame([reminder_row])
    rem_df.to_csv(REMINDERS_FILE, index=False)

def get_reminders_for_user(username):
    if os.path.exists(REMINDERS_FILE):
        rem_df = pd.read_csv(REMINDERS_FILE)
        return rem_df[rem_df["username"] == username].sort_values("reminder_date")
    return pd.DataFrame()

# ---------- Patient UI ----------

def patient_page():
    
    
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("← Back to Main", key="back_to_role", use_container_width=True):
            st.session_state.current_page = "role_selection"
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    # --- Login / Signup ---
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        
    if not st.session_state.logged_in:
      
        st.markdown('<div class="centered-container">', unsafe_allow_html=True)
        st.markdown("""
        <div class="login-container">
            <h2 style="color: #2E8B57; margin-bottom: 10px; text-align: center;">👤 Patient Portal</h2>
            <p style="color: #718096; text-align: center; margin-bottom: 30px;">Access your health insights and predictions</p>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 **Login**", "📝 **Signup**"])
        
        with tab1:
            st.markdown("### Welcome Back")
            st.markdown("Sign in to access your health dashboard")
            
            username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pw")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🚀 Login to Dashboard", key="patient_login_btn", use_container_width=True):
                    if username and password:
                        success, msg = login(username, password)
                        if success:
                            st.success(msg)
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please enter both username and password")
            
            with col2:
                if st.button("🔄 Clear", key="clear_login", use_container_width=True):
                    st.rerun()
        
        with tab2:
            st.markdown("### Create Account")
            st.markdown("Join our healthcare platform to start monitoring your health")
            
            username = st.text_input("👤 Choose Username", placeholder="Pick a username", key="signup_user")
            password = st.text_input("🔒 Create Password", type="password", placeholder="Create a password", key="signup_pw")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🌟 Create Account", key="patient_signup_btn", use_container_width=True):
                    if username and password:
                        success, msg = signup(username, password)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error("Please enter both username and password")
            
            with col2:
                if st.button("🔄 Clear", key="clear_signup", use_container_width=True):
                    st.rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        return

   
    username = st.session_state.username
    

    st.markdown(f"""
    <div class="patient-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 style="margin: 0; color: #1b5e20; font-size: 2.2rem;">👤 Welcome, {username}</h1>
                <p style="margin: 0; color: #2e7d32; font-size: 1.2rem;">Your Personal Health Dashboard</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation and logout
    col1, col2, col3 = st.columns([4, 1, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_dashboard", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("🚪 Logout", key="patient_logout", use_container_width=True):
            st.session_state.current_page = "role_selection"
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown("---")
    
    # Navigation menu with clear radio buttons
    st.markdown("### 📍 Navigation")
    menu_option = st.radio(
        "Choose a section:",
        ["📤 Upload Report", "📊 View Reports", "🤖 AI Insights", "⏰ Reminders"],
        horizontal=True,
        key="patient_navigation"
    )
    
    st.markdown("---")
    
    # OPTION 1: Upload Medical Report
    if menu_option == "📤 Upload Report":
        st.markdown("## 📤 Upload Medical Report")
        st.markdown("Upload your medical test results to get AI-powered health insights and risk predictions.")
        
        st.markdown("""
        <div class="upload-area">
            <h3 style="color: #2E8B57; margin-bottom: 15px;">📁 Upload Your Medical Report</h3>
            <p style="color: #718096; font-size: 1.1rem; margin-bottom: 5px;">Supported formats: CSV, Excel</p>
            <p style="color: #a0aec0; font-size: 1rem;">Drag and drop your file here or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Choose your medical report file", type=["csv", "xlsx"], label_visibility="collapsed", key="file_uploader")
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"❌ Failed to parse uploaded file: {e}")
                return

            save_patient_report(username, df)
            st.success("✅ Report saved successfully!")
            
            # Get latest report and show ML predictions immediately
            patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
            if os.path.exists(patient_file):
                all_reports = pd.read_csv(patient_file)
                if "timestamp" in all_reports.columns:
                    try:
                        all_reports["timestamp_parsed"] = pd.to_datetime(all_reports["timestamp"])
                        all_reports = all_reports.sort_values("timestamp_parsed", ascending=False)
                    except Exception:
                        all_reports = all_reports.sort_values("timestamp", ascending=False)
                
                latest_row = all_reports.head(1).squeeze()
                
                st.markdown("## 🔬 Health Risk Assessment")
                
                # Diabetes prediction
                
                if all(col in all_reports.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]):
                    try:
                        label, prob = predict_diabetes(latest_row.to_frame().T)
                        risk_class = "risk-high" if prob >= 0.7 else "risk-moderate" if prob >= 0.3 else "risk-low"
                        risk_text = "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.3 else "LOW"
                        risk_color = "🔴" if prob >= 0.7 else "🟡" if prob >= 0.3 else "🟢"
                        risk_bg_color = "#ff4444" if prob >= 0.7 else "#ffaa00" if prob >= 0.3 else "#44ff44"
                        
                        st.markdown(f"""
                        <div class="metric-card {risk_class}">
                            <h3 style="color: #2d3748; margin-bottom: 20px; text-align: center;">🩺 Diabetes Risk Assessment</h3>
                            <div style="text-align: center; padding: 20px; background: {risk_bg_color}20; border-radius: 15px; border: 2px solid {risk_bg_color};">
                                <p style="font-size: 2rem; margin: 10px 0; font-weight: 800; color: {risk_bg_color};">
                                    {risk_color} {risk_text} RISK
                                </p>
                                <p style="font-size: 1.5rem; margin: 5px 0; font-weight: 700; color: #2d3748;">
                                    Probability: {prob:.2f}
                                </p>
                                <p style="font-size: 1rem; margin: 10px 0 0 0; color: #718096; font-weight: 500;">
                                    {'⚠️ Immediate medical attention recommended' if prob >= 0.7 else '🔄 Regular monitoring advised' if prob >= 0.3 else '✅ Low risk - maintain healthy lifestyle'}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"ℹ️ Diabetes prediction not available: {e}")
                
                # Heart disease prediction
                if all(col in all_reports.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]):
                    try:
                        label, prob = predict_heart(latest_row.to_frame().T)
                        risk_class = "risk-high" if prob >= 0.7 else "risk-moderate" if prob >= 0.3 else "risk-low"
                        risk_text = "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.3 else "LOW"
                        risk_color = "🔴" if prob >= 0.7 else "🟡" if prob >= 0.3 else "🟢"
                        risk_bg_color = "#ff4444" if prob >= 0.7 else "#ffaa00" if prob >= 0.3 else "#44ff44"
                        
                        st.markdown(f"""
                        <div class="metric-card {risk_class}">
                            <h3 style="color: #2d3748; margin-bottom: 20px; text-align: center;">❤️ Heart Disease Risk Assessment</h3>
                            <div style="text-align: center; padding: 20px; background: {risk_bg_color}20; border-radius: 15px; border: 2px solid {risk_bg_color};">
                                <p style="font-size: 2rem; margin: 10px 0; font-weight: 800; color: {risk_bg_color};">
                                    {risk_color} {risk_text} RISK
                                </p>
                                <p style="font-size: 1.5rem; margin: 5px 0; font-weight: 700; color: #2d3748;">
                                    Probability: {prob:.2f}
                                </p>
                                <p style="font-size: 1rem; margin: 10px 0 0 0; color: #718096; font-weight: 500;">
                                    {'⚠️ Immediate medical attention recommended' if prob >= 0.7 else '🔄 Regular monitoring advised' if prob >= 0.3 else '✅ Low risk - maintain healthy lifestyle'}
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"ℹ️ Heart disease prediction not available: {e}")
                
                # Show key metrics visualization 
                st.markdown("### 📈 Key Health Metrics")
                numeric_cols = ["Glucose", "BMI", "Cholesterol", "BloodPressure", "MaxHR"]
                available_cols = [c for c in numeric_cols if c in all_reports.columns]

                if len(available_cols) > 0:
                    values = []
                    labels = []
                    for c in available_cols:
                        try:
                            val = float(latest_row.get(c, np.nan))
                        except Exception:
                            val = np.nan
                        labels.append(c)
                        values.append(val)

                    fig, ax = plt.subplots(figsize=(12, 6))
                    colors = ['#2E8B57', '#3CB371', '#48BB78', '#68D391', '#9AE6B4']
                    bars = ax.bar(labels, values, color=colors, alpha=0.8)
                    ax.set_facecolor('#f8fff9')
                    fig.patch.set_facecolor('#f8fff9')
                    
                    # Add value labels on bars
                    for i, v in enumerate(values):
                        if pd.notna(v):
                            ax.text(i, v + max(0.1, 0.01 * abs(v)), f"{v:.1f}", 
                                ha='center', fontweight='bold', fontsize=11)
                    
                    numeric_vals = [v for v in values if not pd.isna(v)]
                    if numeric_vals:
                        ax.set_ylim(0, max(numeric_vals) * 1.3)
                    
                    ax.set_xlabel("Health Metrics", fontsize=12, fontweight='bold', color='#2d3748')
                    ax.set_ylabel("Values", fontsize=12, fontweight='bold', color='#2d3748')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                

    # OPTION 2: View Reports & Health Trends
    elif menu_option == "📊 View Reports":
        st.markdown("## 📊 View Reports & Health Trends")
        st.markdown("Monitor your health data over time and track your risk factors.")
        
        patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
        if not os.path.exists(patient_file):
            st.info("📝 No reports found yet. Upload a report to get started.")
            return

        all_reports = pd.read_csv(patient_file)

        if "timestamp" in all_reports.columns:
            try:
                all_reports["timestamp_parsed"] = pd.to_datetime(all_reports["timestamp"])
                all_reports = all_reports.sort_values("timestamp_parsed", ascending=False)
            except Exception:
                all_reports = all_reports.sort_values("timestamp", ascending=False)
        else:
            all_reports = all_reports.sort_values(all_reports.columns[0], ascending=False)

        st.markdown("### 📋 Recent Medical Reports")
        st.dataframe(get_last_reports(username, n=5), use_container_width=True)

        # Compute predictions
        def safe_predict_diabetes_row(row):
            try:
                lbl, p = predict_diabetes(row.to_frame().T)
                return p
            except Exception:
                return np.nan

        def safe_predict_heart_row(row):
            try:
                lbl, p = predict_heart(row.to_frame().T)
                return p
            except Exception:
                return np.nan

        if "diabetes_prob" not in all_reports.columns:
            all_reports["diabetes_prob"] = all_reports.apply(safe_predict_diabetes_row, axis=1)
        else:
            all_reports["diabetes_prob"] = pd.to_numeric(all_reports["diabetes_prob"], errors="coerce")

        if "heart_prob" not in all_reports.columns:
            all_reports["heart_prob"] = all_reports.apply(safe_predict_heart_row, axis=1)
        else:
            all_reports["heart_prob"] = pd.to_numeric(all_reports["heart_prob"], errors="coerce")

        st.markdown("### 📉 Health Risk Trends Over Time")
        try:
            idx = "timestamp_parsed" if "timestamp_parsed" in all_reports.columns else "timestamp"
            chart_df = all_reports[[idx, "diabetes_prob", "heart_prob"]].set_index(idx)
            st.line_chart(chart_df)
        except Exception:
            st.info("📊 Not enough data to plot risk trends yet. Upload more reports to see trends.")

        # Risk Level Indicators
        latest = all_reports.iloc[0]
        st.markdown("### 🚦 Current Risk Levels")

        def safe_risk_display(prob, label):
            if pd.isna(prob):
                st.info(f"ℹ️ No prediction available for {label}")
                return

            if prob < 0.3:
                st.success(f"🟢 **{label} Risk: LOW** (Probability: {prob:.2f})")
            elif prob < 0.7:
                st.warning(f"🟡 **{label} Risk: MODERATE** (Probability: {prob:.2f})")
            else:
                st.error(f"🔴 **{label} Risk: HIGH** (Probability: {prob:.2f})")

        col1, col2 = st.columns(2)
        with col1:
            safe_risk_display(latest.get("diabetes_prob", np.nan), "Diabetes")
        with col2:
            safe_risk_display(latest.get("heart_prob", np.nan), "Heart Disease")

        # Critical Alerts
        st.markdown("### ⚠️ Critical Alerts")
        critical_seen = False
        critical_rules = [
            ("Glucose", lambda v: pd.notna(v) and v >= 300, "Very high blood glucose (>=300). Seek urgent medical attention."),
            ("BloodPressure", lambda v: pd.notna(v) and v >= 180, "Hypertensive crisis (systolic >=180). Seek immediate care."),
            ("Cholesterol", lambda v: pd.notna(v) and v >= 300, "Very high cholesterol (>=300). Contact your physician."),
            ("MaxHR", lambda v: pd.notna(v) and v <= 35, "Very low maximum heart rate recorded; consult doctor."),
        ]
        for col, rule_fn, msg in critical_rules:
            if col in latest.index:
                try:
                    val = float(latest.get(col, np.nan))
                except Exception:
                    val = np.nan
                if rule_fn(val):
                    st.error(f"⚠️ **{col}: {val}** — {msg}")
                    critical_seen = True

        if not critical_seen:
            st.success("✅ No critical alerts detected in your latest report")
    
    # OPTION 3: AI Insights & Health Summary
    elif menu_option == "🤖 AI Insights":
        st.markdown("## 🤖 AI Insights & Health Summary")
        st.markdown("Get personalized health insights and recommendations powered by artificial intelligence.")
        
        patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
        if not os.path.exists(patient_file):
            st.info("📝 No reports found yet. Upload a report to get AI insights.")
            return

        all_reports = pd.read_csv(patient_file)
        if "timestamp" in all_reports.columns:
            try:
                all_reports["timestamp_parsed"] = pd.to_datetime(all_reports["timestamp"])
                all_reports = all_reports.sort_values("timestamp_parsed", ascending=False)
            except Exception:
                all_reports = all_reports.sort_values("timestamp", ascending=False)

        st.markdown("### 💡 AI-Generated Health Insights")
        with st.spinner("🔄 Generating personalized insights..."):
            try:
                insights = generate_patient_insights(all_reports)
                
                with st.container():
                    st.markdown(
                        f"""
                        <div style='
                            background: white;
                            color: #2d3748;
                            padding: 30px;
                            border-radius: 15px;
                            margin: 20px 0;
                            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                            border: 1px solid #e9ecef;
                            line-height: 1.6;
                            font-size: 1.05rem;
                        '>
                        """, 
                        unsafe_allow_html=True
                    )
                    st.markdown(insights)
                    st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Failed to generate AI insights: {e}")

        st.markdown("### 📥 Download Health Summary")
        st.markdown("Download a comprehensive summary of your health report including AI insights and predictions.")
        
        latest_row = all_reports.head(1).squeeze()
        latest_for_summary = latest_row.to_dict()
        latest_preds = {
            "diabetes_prob": latest_row.get("diabetes_prob", np.nan) if "diabetes_prob" in latest_row.index else np.nan,
            "heart_prob": latest_row.get("heart_prob", np.nan) if "heart_prob" in latest_row.index else np.nan
        }
        summary_bytes, mime, filename = create_health_summary_bytes(username, latest_for_summary, latest_preds, insights)

        st.download_button(
            "📄 Download Health Summary Report", 
            data=summary_bytes, 
            file_name=filename, 
            mime=mime, 
            use_container_width=True,
            help="Download a comprehensive PDF report of your health data and insights"
        )
    
    # OPTION 4: Manage Reminders
    elif menu_option == "⏰ Reminders":
        st.markdown("## ⏰ Manage Health Reminders")
        st.markdown("Set reminders for medication, doctor appointments, and health check-ups.")
        
        st.markdown("### ➕ Set New Reminder")
        col_date, col_note = st.columns([1, 2])
        with col_date:
            remind_date = st.date_input("Reminder Date", help="Select the date for your reminder")
        with col_note:
            remind_note = st.text_input("Reminder Note", placeholder="e.g., Doctor appointment, Medication refill, Lab test", 
                                      help="Add a note to remember what this reminder is for")
        
        if st.button("💾 Save Reminder", use_container_width=True):
            if remind_note.strip():
                save_reminder(username, remind_date.strftime("%Y-%m-%d"), remind_note)
                st.success("✅ Reminder saved successfully!")
                st.rerun()
            else:
                st.warning("⚠️ Please add a note for your reminder")

        st.markdown("### 📋 Your Upcoming Reminders")
        rems = get_reminders_for_user(username)
        if not rems.empty:

            today = datetime.now().strftime("%Y-%m-%d")
            future_rems = rems[rems["reminder_date"] >= today]
            
            if not future_rems.empty:
                for _, reminder in future_rems.iterrows():
                    st.markdown(f"""
                    <div class="reminder-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #2E8B57; font-size: 1.1rem;">📅 {reminder['reminder_date']}</strong>
                                <p style="margin: 5px 0 0 0; color: #4a5568;">{reminder['note']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("📝 No upcoming reminders. Add a new reminder above.")
        else:
            st.info("📝 No reminders set yet. Add your first reminder above.")