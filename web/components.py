import streamlit as st
import pandas as pd

def show_project_name_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Let's start with the basics")
    st.markdown("What would you like to call your forecasting project?")
    
    project_name = st.text_input(
        "",
        placeholder="e.g., Sales Forecast Q1 2024",
        key="project_name_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Next →", disabled=not project_name, use_container_width=True):
            st.session_state.project_data['name'] = project_name
            st.session_state.step = 2
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_uncertainty_type_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 What type of prediction do you need help with?")
    
    uncertainty_type = st.selectbox(
        "",
        options=[
            "📊 Predict future trends (Time-series forecasting)",
            "🎯 Classify data into categories (Coming soon)",
            "🎲 Run what-if scenarios (Coming soon)"
        ],
        key="uncertainty_type"
    )
    
    if "Coming soon" in uncertainty_type:
        st.info("This feature is under development. Currently, only time-series forecasting is available.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Next →", disabled="Coming soon" in uncertainty_type, use_container_width=True):
            st.session_state.project_data['type'] = 'time-series'
            st.session_state.step = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_upload_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Upload your historical data")
    st.markdown("We need dates (YYYY-MM-DD format) and numbers. Here's what your data should look like:")
    
    # Show sample data format
    sample_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22'],
        'value': [100, 105, 98, 110]
    })
    st.dataframe(sample_data, use_container_width=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Your CSV should have 'date' and 'value' columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            
            # Validate data format
            if 'date' not in df.columns or len(df.columns) < 2:
                st.error("❌ Please ensure your CSV has 'date' and at least one value column")
                return
            
            # Show preview
            st.markdown("**Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            # Save the data
            st.session_state.project_data['dataframe'] = df
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("← Back", use_container_width=True):
                    st.session_state.step = 2
                    st.rerun()
            with col3:
                if st.button("Next →", use_container_width=True):
                    st.session_state.step = 4
                    st.rerun()
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.step = 2
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_config_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### ⚙️ How often does your data occur?")
    
    freq_options = {
        "Daily": "D",
        "Weekly (Monday)": "W-MON",
        "Weekly (Sunday)": "W-SUN",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }
    
    selected_freq = st.selectbox(
        "Select your data frequency:",
        options=list(freq_options.keys()),
        index=1  # Default to Weekly (Monday)
    )
    
    st.session_state.project_data['freq'] = freq_options[selected_freq]
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    with col3:
        if st.button("Next →", use_container_width=True):
            st.session_state.step = 5
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_ready_step():
    from forecasting import run_forecasting
    
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 🚀 Ready to see the future?")
    st.markdown("We'll analyze your data using 6 different forecasting methods and show you which works best!")
    
    # Show summary
    st.markdown("**Project Summary:**")
    st.write(f"📝 **Project Name:** {st.session_state.project_data['name']}")
    st.write(f"🎯 **Type:** Time-series forecasting")
    st.write(f"📊 **Data Points:** {len(st.session_state.project_data['dataframe'])} records")
    st.write(f"⚙️ **Frequency:** {st.session_state.project_data['freq']}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 4
            st.rerun()
    with col3:
        if st.button("📊 Start Forecasting!", use_container_width=True, type="primary"):
            run_forecasting()
    
    st.markdown('</div>', unsafe_allow_html=True)