import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

def create_chart_layout(best_model_name):
    """Create chart layout with transparent background"""
    return {
        "title": f"Best Model Performance: {best_model_name}",
        "xaxis_title": "Date",
        "yaxis_title": "Value",
        "yaxis2": dict(
            title="Cumulative Abs % Error",
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 50]
        ),
        "hovermode": 'x unified',
        "height": 600,
        "legend": dict(x=0.02, y=0.98),
        "plot_bgcolor": 'rgba(0,0,0,0)',
        "paper_bgcolor": 'rgba(0,0,0,0)',
        "font": dict(color='#333333'),
        "xaxis": dict(
            gridcolor='rgba(200,200,200,0.3)',
            zerolinecolor='rgba(200,200,200,0.5)'
        ),
        "yaxis": dict(
            gridcolor='rgba(200,200,200,0.3)',
            zerolinecolor='rgba(200,200,200,0.5)'
        )
    }

def generate_future_forecasts(best_model_name, project_name):
    """Generate future forecasts using experimenter"""
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
    
    return future_df

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
            # Generate future forecasts
            future_df = generate_future_forecasts(best_model_name, project_name)
            
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
            
            # Update layout
            fig.update_layout(**create_chart_layout(best_model_name))
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