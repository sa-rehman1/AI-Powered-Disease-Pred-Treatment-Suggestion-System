import streamlit as st
from patient_portal.patient_ui import patient_page
from doctor_portal.doctor_ui import doctor_page

st.set_page_config(page_title="AI Disease Prediction", layout="wide", initial_sidebar_state="collapsed")

# Simple CSS - just background color
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
    }
    
    .main {
        background: transparent;
    }
    
    h1 {
        color: #1b5e20;
        text-align: center;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False
    
    if not st.session_state.logged_in and not st.session_state.doctor_logged_in:
        st.title("🏥 AI Disease Prediction & Treatment System")
        
        st.write("")
        st.subheader("Select Your Role")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            role = st.radio("", ["👤 Patient", "👨‍⚕️ Doctor"], horizontal=True, label_visibility="collapsed")
        
        st.write("")
        
        if role == "👤 Patient":
            patient_page()
        else:
            doctor_page()
    else:
        if st.session_state.logged_in:
            patient_page()
        else:
            doctor_page()

if __name__ == "__main__":
    main()