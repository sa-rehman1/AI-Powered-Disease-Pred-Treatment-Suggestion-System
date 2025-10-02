import streamlit as st
from patient_portal.patient_ui import patient_page
from doctor_portal.doctor_ui import doctor_page

st.set_page_config(
    page_title="AI Disease Prediction", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🏥"
)


st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border-radius: 20px;
        padding: 40px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .role-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    .role-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 30px;
        margin: 15px 0;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
        cursor: pointer;
        text-align: center;
    }
    
    .role-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(46, 139, 87, 0.15);
        border-color: #2E8B57;
    }
    
    .role-card.selected {
        border-color: #2E8B57;
        background: linear-gradient(135deg, #f0fff4 0%, #e6fffa 100%);
    }
    
    .role-icon {
        font-size: 3.5rem;
        margin-bottom: 15px;
        display: block;
    }
    
    .role-title {
        color: #2d3748;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .role-description {
        color: #718096;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        padding: 14px 35px;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4);
        background: linear-gradient(135deg, #267349 0%, #34a865 100%);
    }
    
    /* Fix radio button styling */
    .stRadio > div {
        background: white;
        border-radius: 15px;
        padding: 25px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .stRadio > div > label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #2d3748 !important;
    }
    
    .welcome-title {
        color: white;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    
    /* Remove white strip */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    section.main {
        padding-top: 0px;
    }
    
    /* Make sure everything is visible */
    .css-1d391kg, .css-12oz5g7 {
        background: transparent;
    }
    
    /* Remove extra padding */
    div.stApp > header {
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "role_selection"
    if "selected_role" not in st.session_state:
        st.session_state.selected_role = None
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "doctor_logged_in" not in st.session_state:
        st.session_state.doctor_logged_in = False
    
    # Show appropriate page based on session state
    if st.session_state.current_page == "role_selection":
        show_role_selection()
    elif st.session_state.current_page == "patient_portal":
        patient_page()
    elif st.session_state.current_page == "doctor_portal":
        doctor_page()

def show_role_selection():
    """Show the role selection page"""
    st.markdown("""
    <div class="main-header">
        <h1 class="welcome-title">🏥 AI Disease Prediction & Treatment System</h1>
        <p class="subtitle">Intelligent Healthcare Powered by AI • Choose Your Role to Continue</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Role Selection Container
    st.markdown('<div class="role-container">', unsafe_allow_html=True)
    
    st.markdown("### 👥 Select Your Role")
    st.markdown("Choose how you want to access the system:")
    
  
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        role = st.radio(
            "Choose your role:",
            ["👤 Patient", "👨‍⚕️ Doctor"],
            horizontal=True,
            key="role_selector"
        )
    
   
    col_left, col_right = st.columns(2)
    
    with col_left:
        is_selected = "selected" if role == "👤 Patient" else ""
        st.markdown(f"""
        <div class="role-card {is_selected}">
            <div class="role-icon">👤</div>
            <div class="role-title">Patient</div>
            <div class="role-description">
                Upload medical reports, get AI-powered predictions, track your health trends, and receive personalized insights.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        is_selected = "selected" if role == "👨‍⚕️ Doctor" else ""
        st.markdown(f"""
        <div class="role-card {is_selected}">
            <div class="role-icon">👨‍⚕️</div>
            <div class="role-title">Doctor</div>
            <div class="role-description">
                Monitor patient health data, analyze risk trends, provide medical guidance, and manage patient care.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    

    if st.button(f"🚀 Continue as {'Patient' if role == '👤 Patient' else 'Doctor'}", 
                 use_container_width=True, type="primary"):
        st.session_state.selected_role = role
        if role == "👤 Patient":
            st.session_state.current_page = "patient_portal"
        else:
            st.session_state.current_page = "doctor_portal"
        st.rerun()

if __name__ == "__main__":
    main()