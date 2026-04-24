import streamlit as st
import pandas as pd
import os

def show_uncertainty_type_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 🎯 What type of prediction do you need help with?")
    
    uncertainty_type = st.selectbox(
        "",
        options=[
            "📊 Forecast future values (Time-series forecasting)",
            "🎯 Classify data into categories (Coming soon)",
            "🎲 Test what-if scenarios (Coming soon)"
        ],
        key="uncertainty_type"
    )
    
    if "Coming soon" in uncertainty_type:
        st.info("This feature is under development. Currently, only time-series forecasting is available.")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("Next →", disabled="Coming soon" in uncertainty_type, use_container_width=True):
            st.session_state.project_data['type'] = 'time-series'
            st.session_state.step = 2
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def _configure_data(df, project_name):
    """Shared helper: preview data, pick frequency, store in session state.
    Returns True when configuration is complete."""
    # Validate data format
    if 'date' not in df.columns or len(df.columns) < 2:
        st.error("❌ Please ensure your CSV has 'date' and at least one value column")
        return False

    st.markdown("**Data Preview:**")
    st.dataframe(df.head(), use_container_width=True)

    st.session_state.project_data['name'] = project_name
    st.session_state.project_data['dataframe'] = df

    st.markdown("### ⚙️ How often does your data occur?")

    freq_options = {
        "Daily": ("D", "day"),
        "Weekly (Monday)": ("W-MON", "week"),
        "Weekly (Sunday)": ("W-SUN", "week"),
        "Monthly": ("M", "month"),
        "Quarterly": ("Q", "quarter"),
        "Yearly": ("Y", "year")
    }

    selected_freq = st.selectbox(
        "Select your data frequency:",
        options=list(freq_options.keys()),
        index=1  # Default to Weekly (Monday)
    )

    freq, freq_type = freq_options[selected_freq]
    st.session_state.project_data['freq'] = freq
    st.session_state.project_data['freq_type'] = freq_type
    return True


def show_data_upload_step():
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 📊 Upload and configure your data")
    st.markdown("We need dates (YYYY-MM-DD format) and numbers. Here's what your data should look like:")

    # Show sample data format
    sample_data = pd.DataFrame({
        'date': ['2023-01-01', '2023-01-08', '2023-01-15', '2023-01-22'],
        'value': [100, 105, 98, 110]
    })
    st.dataframe(sample_data, use_container_width=True)

    # --- Demo / Upload toggle ---
    data_source = st.radio(
        "Choose a data source:",
        ["Upload my own CSV", "Try with demo data (TSA Travel)"],
        horizontal=True,
        key="data_source"
    )

    data_configured = False

    if data_source == "Try with demo data (TSA Travel)":
        demo_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "data", "tsa_checkpoint_travel_count_mon_weekly.csv"
        )
        try:
            df = pd.read_csv(demo_path)
            st.info("📂 Using demo dataset: **TSA Checkpoint Travel Numbers** (weekly)")
            st.session_state.project_data['is_demo'] = True
            data_configured = _configure_data(df, "tsa_checkpoint_travel_count_mon_weekly")
        except Exception as e:
            st.error(f"❌ Could not load demo data: {str(e)}")
    else:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Your CSV should have 'date' and 'value' columns"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
                project_name = uploaded_file.name.replace('.csv', '').replace(' ', '_').lower()
                st.session_state.project_data['is_demo'] = False
                data_configured = _configure_data(df, project_name)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with col3:
        if st.button("Next →", disabled=not data_configured, use_container_width=True):
            st.session_state.step = 3
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def show_ready_step():
    from forecasting import run_forecasting
    
    st.markdown('<div class="question-container">', unsafe_allow_html=True)
    st.markdown("### 🚀 Ready to view Forecast?")
    st.markdown("We will take care of the rest. We will be building forecasting model for you!")
    
    # Show summary
    st.markdown("**Project Summary:**")
    st.write(f"📝 **Project Name:** {st.session_state.project_data['name']}")
    st.write(f"🎯 **Type:** Time-series forecasting")
    st.write(f"📊 **Data Points:** {len(st.session_state.project_data['dataframe'])} records")
    st.write(f"⚙️ **Frequency:** {st.session_state.project_data['freq']}")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    with col3:
        if st.button("📊 Start Forecasting!", use_container_width=True, type="primary"):
            run_forecasting()
    
    st.markdown('</div>', unsafe_allow_html=True)