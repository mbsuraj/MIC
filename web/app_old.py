import streamlit as st
import pandas as pd
import json
import os
import sys
import subprocess
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
# Also add project root for src imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set page config
st.set_page_config(
    page_title="MIC - Make It Certain",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern bright appearance
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: #f8f9fa;
    }
    
    /* Main content area */
    .main .block-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Fancy title styling */
    .main-header {
        text-align: center;
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #f5576c);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
        margin-bottom: 1rem;
        font-family: 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Question containers */
    .question-container {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    .metric-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

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
            "🔮 Predict future trends (Time-series forecasting)",
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

def run_forecasting():
    """Prepare data and run the forecasting pipeline"""
    with st.spinner("📊 Running forecasting models... This may take a few minutes."):
        try:
            # Get absolute paths
            web_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(web_dir)
            
            # Create project directory structure
            project_name = st.session_state.project_data['name'].replace(' ', '_').lower()
            
            # Save data to the data directory
            data_dir = os.path.join(project_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            data_path = os.path.join(data_dir, f"{project_name}.csv")
            st.session_state.project_data['dataframe'].to_csv(data_path, index=False)
            
            # Create/update config
            config_dir = os.path.join(project_root, "config")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "data_config.json")
            
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            config[project_name] = {
                "format": "%Y-%m-%d",
                "freq": st.session_state.project_data['freq']
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Run the main forecasting script
            main_script = os.path.join(project_root, "src", "main.py")
            
            # Set PYTHONPATH to include the project root so imports work
            env = os.environ.copy()
            env['PYTHONPATH'] = project_root
            
            result = subprocess.run([
                sys.executable, main_script
            ], capture_output=True, text=True, cwd=project_root, env=env)
            
            if result.returncode == 0:
                st.session_state.step = 6
                st.success("✅ Forecasting completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error running forecasting: {result.stderr}")
                st.error(f"❌ Stdout: {result.stdout}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_results_dashboard():
    st.markdown("### 📈 Forecasting Results Dashboard")
    
    try:
        # Load results
        web_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(web_dir)
        
        training_results = pd.read_csv(os.path.join(project_root, "output", "training_results.csv"))
        testing_results = pd.read_csv(os.path.join(project_root, "output", "testing_results.csv"))
        
        # Find best model by MAPE
        best_model_row = testing_results.loc[testing_results['mape'].idxmin()]
        best_model_name = best_model_row['model_name']
        
        # Display best model info with explanations
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏆 Best Model", best_model_name)
        with col2:
            st.metric("📊 MAPE", f"{best_model_row['mape']:.3f}")
            with st.expander("ℹ️ What is MAPE?"):
                st.write("**Mean Absolute Percentage Error** - Average percentage difference between actual and predicted values. Lower is better. 5% = very good, 10% = good, 20% = acceptable.")
        with col3:
            st.metric("📈 SMAPE", f"{best_model_row['smape']:.3f}")
            with st.expander("ℹ️ What is SMAPE?"):
                st.write("**Symmetric Mean Absolute Percentage Error** - Similar to MAPE but handles zero values better. Lower is better. Range: 0-100%.")
        
        # Load forecast data for best model
        project_name = st.session_state.project_data['name'].replace(' ', '_').lower()
        
        # Try both parametric and non-parametric folders
        best_forecast_data = None
        for folder in ['parametric', 'non_parametric']:
            try:
                forecast_path = os.path.join(project_root, "output", "forecasts", folder, f"{project_name}.csv")
                forecast_data = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
                if best_model_name in forecast_data.columns:
                    best_forecast_data = forecast_data[['y_true', best_model_name]].copy()
                    break
            except:
                continue
        
        if best_forecast_data is not None:
            # Generate future forecasts using experimenter
            future_df = None
            with st.spinner("🔄 Generating future forecasts..."):
                try:
                    # Temporarily add project root to sys.path for imports
                    web_dir = os.path.dirname(os.path.abspath(__file__))
                    project_root = os.path.dirname(web_dir)
                    
                    if project_root not in sys.path:
                        sys.path.insert(0, project_root)
                    
                    # Now import experimenter
                    from src.common.experimenter import Experimenter
                    experimenter = Experimenter()

                    # Load datasets and forecasters to populate the experimenter
                    experimenter.load_datasets()
                    experimenter.get_full_data()
                    experimenter.load_forecasters()
                    
                    # Get the dataset name for the best model
                    dataset_name = project_name  # This should match the dataset name used in the experiment
                    
                    # Generate future forecasts for specific dataset and model
                    future_df = experimenter.retrain_best_model_and_forecast_future(best_model_name, dataset_name, periods=52)
                    if not future_df.empty:
                        st.success("✅ Future forecasts generated!")
                    else:
                        st.warning(f"Could not generate future forecasts - method returned None for model: {best_model_name}")
                        
                except Exception as e:
                    st.error(f"Error generating future forecasts: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
            
            # Calculate cumulative absolute percentage error at each time period
            y_true = best_forecast_data['y_true']
            y_pred = best_forecast_data[best_model_name]
            
            # Cumulative absolute percentage error (capped at 100%)
            abs_pct_errors = np.abs((y_true - y_pred) / y_true) * 100
            abs_pct_errors = np.clip(abs_pct_errors, 0, 100)  # Cap between 0-100%
            cumulative_errors = abs_pct_errors.cumsum() / np.arange(1, len(abs_pct_errors) + 1)
            
            # Create dual-axis chart
            fig = go.Figure()
            
            # Add actual vs predicted on primary axis
            fig.add_trace(go.Scatter(
                x=best_forecast_data.index,
                y=y_true,
                mode='lines',
                name='Actual',
                line=dict(color='black', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=best_forecast_data.index,
                y=y_pred,
                mode='lines',
                name=f'{best_model_name} Forecast',
                line=dict(color='blue', width=2, dash='dash')
            ))
            
            # Add cumulative error on secondary axis
            fig.add_trace(go.Scatter(
                x=best_forecast_data.index,
                y=cumulative_errors,
                mode='lines',
                name='Cumulative Abs % Error',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
            
            # Add future forecasts if available
            if future_df is not None:
                fig.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df.iloc[:, 0],
                    mode='lines',
                    name='Future Forecast',
                    line=dict(color='green', width=2, dash='dot')
                ))
            
            # Update layout for dual axis
            fig.update_layout(
                title=f"Best Model Performance: {best_model_name}",
                xaxis_title="Date",
                yaxis_title="Value",
                yaxis2=dict(
                    title="Cumulative Abs % Error",
                    overlaying='y',
                    side='right',
                    showgrid=False,
                    range=[0, 50]
                ),
                hovermode='x unified',
                height=600,
                legend=dict(x=0.02, y=0.98),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#333333'),
                xaxis=dict(
                    gridcolor='rgba(200,200,200,0.3)',
                    zerolinecolor='rgba(200,200,200,0.5)'
                ),
                yaxis=dict(
                    gridcolor='rgba(200,200,200,0.3)',
                    zerolinecolor='rgba(200,200,200,0.5)'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            export_data = best_forecast_data.copy()
            export_data['cumulative_abs_pct_error'] = cumulative_errors
            
            # Add future forecasts to export if available
            if future_df is not None:
                export_data = pd.concat([export_data, future_df])
            
            export_data.index.name = 'date'
            
            csv_data = export_data.to_csv()
            st.download_button(
                label="📥 Download Best Model Results (CSV)",
                data=csv_data,
                file_name=f"{project_name}_{best_model_name}_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        else:
            st.error("Could not load forecast data for the best model")
        
        # Option to start new project
        if st.button("🔄 Start New Project", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error loading results: {str(e)}")
        if st.button("🔄 Start New Project", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()