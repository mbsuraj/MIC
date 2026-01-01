import streamlit as st
import pandas as pd
import json
import os
import sys
import subprocess

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
                st.session_state.step = 4
                st.success("✅ Forecasting completed successfully!")
                st.rerun()
            else:
                st.error(f"❌ Error running forecasting: {result.stderr}")
                st.error(f"❌ Stdout: {result.stdout}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")