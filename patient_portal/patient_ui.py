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
    
    # --- Login / Signup ---
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Signup"])
        with tab1:
            st.markdown('<h3 style="color: #1b5e20; text-align: center;">Patient Login</h3>', unsafe_allow_html=True)
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pw", placeholder="Enter your password")
            if st.button("Login", key="patient_login_btn"):
                success, msg = login(username, password)
                if success:
                    st.success(msg)
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error(msg)
        with tab2:
            st.markdown('<h3 style="color: #1b5e20; text-align: center;">Create Account</h3>', unsafe_allow_html=True)
            username = st.text_input("New Username", key="signup_user", placeholder="Choose a username")
            password = st.text_input("New Password", type="password", key="signup_pw", placeholder="Choose a password")
            if st.button("Signup", key="patient_signup_btn"):
                success, msg = signup(username, password)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        return

    username = st.session_state.username
    
    # Welcome header with logout
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f'<h2 style="color: #1b5e20;">👤 Welcome, {username}</h2>', unsafe_allow_html=True)
    with col2:
        if st.button("Logout", key="patient_logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
    
    st.markdown("---")
    
    # Navigation menu
    menu_option = st.radio(
        "Navigation:",
        ["📤 Upload Report", "📊 View Reports", "🤖 AI Insights", "⏰ Reminders"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # OPTION 1: Upload Medical Report
    if menu_option == "📤 Upload Report":
        st.subheader("📤 Upload Medical Report")
        
        uploaded = st.file_uploader("Choose your medical report file", type=["csv", "xlsx"])
        if uploaded:
            try:
                if uploaded.name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to parse uploaded file: {e}")
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
                
                st.markdown("### 🔬 ML Predictions for Uploaded Report")
                
                # Diabetes prediction
                if all(col in all_reports.columns for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]):
                    try:
                        label, prob = predict_diabetes(latest_row.to_frame().T)
                        if prob < 0.3:
                            st.success(f"🟢 **Diabetes Risk: LOW** (Probability: {prob:.2f})")
                        elif prob < 0.7:
                            st.warning(f"🟡 **Diabetes Risk: MODERATE** (Probability: {prob:.2f})")
                        else:
                            st.error(f"🔴 **Diabetes Risk: HIGH** (Probability: {prob:.2f})")
                    except Exception as e:
                        st.info(f"Diabetes prediction not available: {e}")
                else:
                    st.info("Diabetes prediction requires: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")
                
                # Heart disease prediction
                if all(col in all_reports.columns for col in ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]):
                    try:
                        label, prob = predict_heart(latest_row.to_frame().T)
                        if prob < 0.3:
                            st.success(f"🟢 **Heart Disease Risk: LOW** (Probability: {prob:.2f})")
                        elif prob < 0.7:
                            st.warning(f"🟡 **Heart Disease Risk: MODERATE** (Probability: {prob:.2f})")
                        else:
                            st.error(f"🔴 **Heart Disease Risk: HIGH** (Probability: {prob:.2f})")
                    except Exception as e:
                        st.info(f"Heart disease prediction not available: {e}")
                else:
                    st.info("Heart disease prediction requires: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope")
                
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

                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#43a047', '#388e3c', '#2e7d32', '#1b5e20', '#66bb6a']
                    sns.barplot(x=labels, y=values, palette=colors, ax=ax)
                    ax.set_facecolor('#f1f8f4')
                    fig.patch.set_facecolor('#f1f8f4')
                    for i, v in enumerate(values):
                        if pd.notna(v):
                            ax.text(i, v + max(0.1, 0.01 * abs(v)), f"{v:.1f}", ha='center', fontweight='bold')
                    numeric_vals = [v for v in values if not pd.isna(v)]
                    if numeric_vals:
                        ax.set_ylim(0, max(numeric_vals) * 1.3)
                    ax.set_xlabel("Health Metrics", fontsize=11, fontweight='bold', color='#1b5e20')
                    ax.set_ylabel("Values", fontsize=11, fontweight='bold', color='#1b5e20')
                    st.pyplot(fig)
    
    # OPTION 2: View Reports & Health Trends
    elif menu_option == "📊 View Reports":
        st.subheader("📊 View Reports & Health Trends")
        
        patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
        if not os.path.exists(patient_file):
            st.info("No reports found yet. Upload a report to get started.")
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

        st.markdown("### 📋 Last 5 Reports")
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

        st.markdown("### 📉 Health Risk Trends")
        try:
            idx = "timestamp_parsed" if "timestamp_parsed" in all_reports.columns else "timestamp"
            chart_df = all_reports[[idx, "diabetes_prob", "heart_prob"]].set_index(idx)
            st.line_chart(chart_df)
        except Exception:
            st.info("Not enough data to plot risk trends.")

        # Risk Level Indicators
        latest = all_reports.iloc[0]
        st.markdown("### 🚦 Current Risk Levels")

        def safe_risk_display(prob, label):
            if pd.isna(prob):
                st.info(f"No prediction available for {label}")
                return

            if prob < 0.3:
                st.success(f"🟢 {label} Risk: **LOW** ({prob:.2f})")
            elif prob < 0.7:
                st.warning(f"🟡 {label} Risk: **MODERATE** ({prob:.2f})")
            else:
                st.error(f"🔴 {label} Risk: **HIGH** ({prob:.2f})")

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
                    st.error(f"⚠️ {col}: {val} — {msg}")
                    critical_seen = True

        if not critical_seen:
            st.success("✅ No critical alerts detected")
    
    # OPTION 3: AI Insights & Health Summary
    elif menu_option == "🤖 AI Insights":
        st.subheader("🤖 AI Insights & Health Summary")
        
        patient_file = os.path.join(PATIENT_DATA_DIR, f"{username}_data.csv")
        if not os.path.exists(patient_file):
            st.info("No reports found yet. Upload a report to get started.")
            return

        all_reports = pd.read_csv(patient_file)
        if "timestamp" in all_reports.columns:
            try:
                all_reports["timestamp_parsed"] = pd.to_datetime(all_reports["timestamp"])
                all_reports = all_reports.sort_values("timestamp_parsed", ascending=False)
            except Exception:
                all_reports = all_reports.sort_values("timestamp", ascending=False)

        st.markdown("### 💡 AI-Generated Insights")
        try:
            insights = generate_patient_insights(all_reports)
            st.markdown(insights)
        except Exception as e:
            st.error(f"Failed to generate AI insights: {e}")
            insights = "AI insights unavailable."

        st.markdown("### 📥 Download Health Summary")
        
        latest_row = all_reports.head(1).squeeze()
        latest_for_summary = latest_row.to_dict()
        latest_preds = {
            "diabetes_prob": latest_row.get("diabetes_prob", np.nan) if "diabetes_prob" in latest_row.index else np.nan,
            "heart_prob": latest_row.get("heart_prob", np.nan) if "heart_prob" in latest_row.index else np.nan
        }
        summary_bytes, mime, filename = create_health_summary_bytes(username, latest_for_summary, latest_preds, insights)

        st.download_button("📄 Download Summary", data=summary_bytes, file_name=filename, mime=mime, use_container_width=True)
    
    # OPTION 4: Manage Reminders
    elif menu_option == "⏰ Reminders":
        st.subheader("⏰ Manage Reminders")
        
        st.markdown("### ➕ Set New Reminder")
        col_date, col_note = st.columns([1, 2])
        with col_date:
            remind_date = st.date_input("Date")
        with col_note:
            remind_note = st.text_input("Note (optional)", placeholder="e.g., Doctor appointment")
        
        if st.button("Save Reminder", use_container_width=True):
            save_reminder(username, remind_date.strftime("%Y-%m-%d"), remind_note)
            st.success("✅ Reminder saved!")
            st.rerun()

        st.markdown("### 📋 Your Reminders")
        rems = get_reminders_for_user(username)
        if not rems.empty:
            st.dataframe(rems, use_container_width=True)
        else:
            st.info("No reminders set yet.")