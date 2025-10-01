import streamlit as st
from patient_portal.patient_ui import patient_page
from doctor_portal.doctor_ui import doctor_page

st.set_page_config(page_title="AI Disease Prediction", layout="wide")

def main():
    st.title("AI Disease Prediction & Treatment Suggestion System")
    st.sidebar.title("Navigation")
    
    # Sidebar radio for page selection
    page = st.sidebar.radio("Go to", ("Patient", "Doctor"))

    if page == "Patient":
        patient_page()
    else:
        doctor_page()

if __name__ == "__main__":
    main()
