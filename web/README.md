# MIC Web Interface

A user-friendly web interface for the MIC (Make It Certain) forecasting framework.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the web interface:**
   ```bash
   ./run.sh
   ```
   
   Or manually:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser to:** `http://localhost:8501`

## Features

- **Google-like Interface**: Clean, intuitive design
- **Step-by-step Workflow**: Guided process for forecasting projects
- **Interactive Dashboard**: Visualize results with Plotly charts
- **Model Comparison**: Compare 6 different forecasting models
- **Export Results**: Download forecasts and performance metrics

## Usage Flow

1. **Project Name**: Give your forecasting project a name
2. **Prediction Type**: Select time-series forecasting
3. **Data Upload**: Upload CSV with date and value columns
4. **Data Configuration**: Set your data frequency (daily, weekly, etc.)
5. **Run Forecasting**: Execute the MIC backend pipeline
6. **View Results**: Interactive dashboard with model comparisons

## Data Format

Your CSV file should have:
- `date` column in YYYY-MM-DD format
- One or more numeric value columns

Example:
```csv
date,value
2023-01-01,100
2023-01-08,105
2023-01-15,98
```

## Technical Details

- Built with Streamlit for rapid development
- Integrates with existing MIC backend
- Uses Plotly for interactive visualizations
- Automatically manages data and configuration files