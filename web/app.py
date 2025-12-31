import streamlit as st
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Also add project root for src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import local modules
from styles import MAIN_STYLES
from components import (
    show_project_name_step,
    show_uncertainty_type_step, 
    show_data_upload_step,
    show_data_config_step,
    show_ready_step
)
from dashboard import show_results_dashboard

# Set page config
st.set_page_config(
    page_title="MIC - Make It Certain",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply styles
st.markdown(MAIN_STYLES, unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 MIC - Make It Certain</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.3rem; color: #555; font-weight: 500;">Your intelligent forecasting companion</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'project_data' not in st.session_state:
        st.session_state.project_data = {}
    
    # Progress bar
    progress = min((st.session_state.step - 1) / 5, 1.0)  # 5 steps total, cap at 1.0
    st.progress(progress)
    
    # Step-by-step questions
    if st.session_state.step == 1:
        show_project_name_step()
    elif st.session_state.step == 2:
        show_uncertainty_type_step()
    elif st.session_state.step == 3:
        show_data_upload_step()
    elif st.session_state.step == 4:
        show_data_config_step()
    elif st.session_state.step == 5:
        show_ready_step()
    elif st.session_state.step == 6:
        show_results_dashboard()

if __name__ == "__main__":
    main()