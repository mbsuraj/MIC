import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys

@st.cache_data
def cached_generate_future_forecasts(best_model_name, project_name):
    """Cached version of future forecast generation"""
    return generate_future_forecasts(best_model_name, project_name)

@st.cache_data
def get_best_model_info():
    """Get best model information from testing results"""
    web_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(web_dir)
    project_name = st.session_state.project_data['name'].replace(' ', '_').lower()
    testing_results = pd.read_csv(os.path.join(project_root, "output", "testing_results.csv"))
    best_model_row = testing_results.loc[testing_results['weighted_mape'].idxmin()]
    model_path = os.path.join(project_root, "cache", f"{project_name}_{best_model_row['model_name']}.pkl")
    with open(model_path, 'rb') as f:
        best_model = pickle.load(f)

    return best_model_row['model_name'], best_model_row['weighted_mape']

def create_chart_layout(mape_score):
    """Create chart layout with transparent background"""
    return {
        "title": f"Forecast Performance",
        "xaxis_title": "Date",
        "yaxis_title": "Value",
        "yaxis2": dict(
            title="Cumulative Abs % Error",
            overlaying='y',
            side='right',
            showgrid=False,
            range=[0, 50]
        ),
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        "hovermode": 'x unified',
        "height": 600,
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
        ),
        "annotations": [
            dict(
                x=0.02, y=0.02,
                xref="paper", yref="paper",
                text=f"% Error: {mape_score:.3f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        ]
    }

def generate_future_forecasts(best_model_name, project_name):
    """Generate future forecasts using experimenter"""
    future_df = None
    with st.spinner("🔄 Generating forecasts..."):
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
                st.success("✅ Forecasts generated!")
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
        best_model_name, mape_score = get_best_model_info()

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
            future_df = cached_generate_future_forecasts(best_model_name, project_name)

            # Create combined dataset for date range selection
            combined_data = best_forecast_data.copy()
            if future_df is not None:
                combined_data = pd.concat([combined_data, future_df])

            # Date range slider
            if future_df is not None and 'confidence' in future_df.columns:
                # Default: middle of training to medium confidence end
                training_mid = best_forecast_data.index[len(best_forecast_data) // 2]
                medium_end = future_df[future_df['confidence'].isin(['high', 'medium'])].index[-1]

                date_range = st.slider(
                    "Select forecast period:",
                    min_value=combined_data.index.min().date(),
                    max_value=combined_data.index.max().date(),
                    value=(training_mid.date(), medium_end.date()),
                    format="YYYY-MM-DD"
                )

                # Filter data based on date range
                start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                display_forecast_data = best_forecast_data[
                    (best_forecast_data.index >= start_date) &
                    (best_forecast_data.index <= end_date)
                    ]
                display_future_df = future_df[
                    (future_df.index >= start_date) &
                    (future_df.index <= end_date)
                    ] if future_df is not None else None
            else:
                display_forecast_data = best_forecast_data
                display_future_df = future_df

            # Update chart creation to use display_forecast_data instead of best_forecast_data
            y_true = display_forecast_data['y_true']
            y_pred = display_forecast_data[best_model_name]

            # Cumulative absolute percentage error (capped at 100%)
            abs_pct_errors = np.abs((y_true - y_pred) / y_true) * 100
            abs_pct_errors = np.clip(abs_pct_errors, 0, 100)  # Cap between 0-100%
            cumulative_errors = abs_pct_errors.cumsum() / np.arange(1, len(abs_pct_errors) + 1)
            
            # Create dual-axis chart
            fig = go.Figure()
            
            # Add actual vs predicted on primary axis
            fig.add_trace(go.Scatter(
                x=display_forecast_data.index,
                y=y_true,
                mode='lines',
                name='Actual',
                opacity=0.5,
                line=dict(color='grey', width=2.5)
            ))
            
            fig.add_trace(go.Scatter(
                x=display_forecast_data.index,
                y=y_pred,
                mode='lines',
                name=f'Model Fit',
                opacity=0.7,
                line=dict(color='#E5E7EB', width=3.5)
            ))
            
            # Add future forecasts if available
            if display_future_df is not None and not display_future_df.empty:
                fig.add_trace(go.Scatter(
                    x=display_future_df.index,
                    y=display_future_df.iloc[:, 0],
                    mode='lines',
                    name='Future Forecast',
                    opacity=0.7,
                    line=dict(color='#E5E7EB', width=3.5, dash='dash')
                ))

                if 'confidence' in display_future_df.columns:
                    confidence_colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
                    for conf_level in ['high', 'medium', 'low']:
                        mask = display_future_df['confidence'] == conf_level
                        if mask.any():
                            fig.add_trace(go.Scatter(
                                x=display_future_df.index[mask],
                                y=display_future_df.iloc[:, 0][mask],
                                mode='markers',
                                name=f'{conf_level.title()} Confidence',
                                marker=dict(color=confidence_colors[conf_level], size=8)
                            ))

            # Update layout

            fig.update_layout(**create_chart_layout(mape_score))
            st.plotly_chart(fig, use_container_width=True)
            
            # Export option
            export_data = display_forecast_data.copy()
            if display_future_df is not None:
                export_data = pd.concat([export_data, display_future_df])

            export_data.index.name = 'date'

            csv_data = export_data.rename(columns={
                'y_true': 'actual',
                f'{best_model_name}': 'model_fit',
                f'{best_model_name}_future': 'forecast'
            }).loc[:, ['actual', 'model_fit', 'forecast', 'confidence']].to_csv()

            st.download_button(
                label="📥 Download Best Model Results (CSV)",
                data=csv_data,
                file_name=f"{project_name}_results.csv",
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