# MIC (Make It Certain)

A comprehensive time series forecasting framework that implements multiple parametric and non-parametric forecasting models with automated hyperparameter tuning and performance evaluation. Includes both a Python backend and an interactive Streamlit web interface.

## Overview

MIC is designed to provide certainty in time series predictions by comparing multiple forecasting approaches across various datasets. The framework automatically handles data preprocessing, model training, hyperparameter optimization, and performance evaluation. A user-friendly web interface allows non-technical users to run forecasting experiments interactively.

## Features

- **Multiple Forecasting Models**: Supports 12+ parametric and non-parametric approaches
- **Automated Data Processing**: Handles data loading, cleaning, and preprocessing with scaling
- **Hyperparameter Optimization**: Randomized search for optimal model parameters
- **Performance Evaluation**: Comprehensive metrics including MAPE, SMAPE, weighted MAPE, and error statistics
- **Model Persistence**: Save and load trained models for reuse
- **Batch Processing**: Process multiple datasets simultaneously
- **Results Export**: Detailed training and testing results with forecasts
- **Web Interface**: Interactive Streamlit dashboard for easy access
- **Interactive Visualizations**: Plotly charts for model comparison and results visualization

## Supported Models

### Parametric Models
- **ARIMA**: AutoRegressive Integrated Moving Average with seasonal components
- **SARIMA**: Seasonal ARIMA
- **ETS**: Exponential Smoothing State Space models
- **Holt-Winters**: Classical Holt-Winters exponential smoothing
- **AR**: AutoRegressive models
- **Bayesian Forecaster**: Bayesian time series modeling
- **Bayesian SSM**: Bayesian State Space Models
- **Support Vector**: SVR-based forecasting

### Non-Parametric Models
- **Gradient Boosting**: LightGBM-based forecasting with lag features
- **Prophet**: Facebook's Prophet for time series with trend and seasonality
- **Neural Networks**: Neural network-based forecasting
- **Regression Trees**: Tree-based forecasting

## Project Structure

```
MIC/
├── src/                          # Python backend
│   ├── main.py                  # Entry point for backend execution
│   ├── common/
│   │   ├── experimenter.py      # Main experiment orchestrator
│   │   ├── dataGenerator.py     # Data loading and preprocessing
│   │   ├── dataPreprocessor.py  # Data scaling and transformation
│   │   ├── forecaster.py        # Abstract base class for models
│   │   └── utils.py             # Evaluation metrics
│   ├── models/                  # Individual forecasting model implementations
│   │   ├── ARIMAForecaster.py
│   │   ├── SARIMAForecaster.py
│   │   ├── ETSForecaster.py
│   │   ├── HoltWintersForecaster.py
│   │   ├── ARForecaster.py
│   │   ├── BayesianForecaster.py
│   │   ├── BayesianSSMForecaster.py
│   │   ├── SVForecaster.py
│   │   ├── GBForecaster.py
│   │   ├── ProphetForecaster.py
│   │   ├── NNForecaster.py
│   │   └── RTForecaster.py
│   ├── pymc_statespace/         # PyMC state space models
│   │   ├── core/                # Core state space functionality
│   │   │   ├── representation.py
│   │   │   └── statespace.py
│   │   ├── filters/             # Kalman filters and smoothers
│   │   │   ├── kalman_filter.py
│   │   │   ├── kalman_smoother.py
│   │   │   ├── distributions.py
│   │   │   └── utilities.py
│   │   ├── models/              # Structural models
│   │   │   ├── structural.py
│   │   │   └── utilities.py
│   │   └── utils/
│   │       ├── constants.py
│   │       └── data_tools.py
│   └── mock_up/                 # Mock implementations and testing
│       └── data_prep.py
├── web/                         # Streamlit web interface
│   ├── app.py                   # Main Streamlit application
│   ├── app_old.py              # Legacy application version
│   ├── components.py            # Reusable UI components
│   ├── dashboard.py             # Results dashboard visualization
│   ├── forecasting.py           # Forecasting workflow logic
│   ├── styles.py                # CSS styling
│   ├── run.sh                   # Script to launch the web interface
│   ├── README.md                # Web interface documentation
│   └── requirements.txt          # Web interface dependencies
├── config/
│   └── data_config.json        # Dataset configuration
├── data/                       # Input datasets
├── cache/                      # Saved trained models
├── output/                     # Results and forecasts
├── requirements.txt            # Python backend dependencies
└── README.md                   # This file
```

## Installation

### Backend and Frontend Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the Web Interface (Recommended)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the web interface:**
   ```bash
   cd web && ./run.sh
   ```
   
   Or manually:
   ```bash
   streamlit run web/app.py
   ```

3. **Open your browser to:** `http://localhost:8501`

4. **Workflow:**
   - Enter project name
   - Select forecasting as prediction type
   - Upload CSV with date and value columns
   - Configure data frequency
   - Run the forecasting pipeline
   - View interactive dashboard with model forecasts

### Using Command Line / Python
1. **Upload Data**  
Input CSV files should contain:
- `date` column in YYYY-MM-DD format
- One numeric value column
- The name of the file should be of format <dataset_name>. 

Example:
```csv
date,value
2023-01-01,100
2023-01-08,105
2023-01-15,98
```
2. **Upload Data Configuration** 
Configure datasets in `config/data_config.json`:

```json
{
  "dataset_name": {
    "format": "%Y-%m-%d",
    "freq": "W-MON",
    "freq_type": "week"
  }
}
```
3. **Run the code**
```commandline
python3.12 src/main.py
```

## Dependencies
Key dependencies include:
- **prophet**: Facebook's Prophet time series library
- **statsmodels**: Statistical modeling package (ARIMA, ETS, Holt-Winters)
- **pymc**: Bayesian modeling library
- **lightgbm**: Gradient boosting framework
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **streamlit**: Web interface framework
- **plotly**: Interactive visualizations
- **scipy**: Scientific computing

See [requirements.txt](requirements.txt) for complete dependency lists.

## Project Components

### Backend (`src/`)

**Experiment Management** (`common/experimenter.py`):
- Orchestrates the entire forecasting pipeline
- Manages dataset loading and splitting
- Handles model training and evaluation
- Exports results

**Data Processing** (`common/dataGenerator.py`, `common/dataPreprocessor.py`):
- Loads data from CSV files
- Applies time series preprocessing
- Handles scaling and normalization
- Supports multiple frequency formats

**Base Forecaster** (`common/forecaster.py`):
- Abstract base class for all forecasting models
- Defines standard interface for fit and predict
- Handles model persistence (save/load)

**Evaluation** (`common/utils.py`):
- Calculates comprehensive metrics
- Supports MAPE, SMAPE, weighted metrics
- Error statistics and comparisons

**Models** (`models/`):
- 12+ individual model implementations
- Each inherits from base Forecaster class
- Customizable hyperparameters
- Support for both point forecasts and intervals

**Advanced State Space Models** (`pymc_statespace/`):
- Kalman filtering and smoothing
- Structural time series models
- Bayesian state space implementations

### Web Interface (`web/`)

**Main Application** (`app.py`):
- Step-by-step workflow interface
- Project management
- Data upload and configuration

**Components** (`components.py`):
- Reusable UI components
- Form inputs and controls
- Configuration panels

**Dashboard** (`dashboard.py`):
- Interactive results visualization
- Model comparison charts
- Performance metrics display

**Forecasting** (`forecasting.py`):
- Integration with backend
- Model execution
- Results processing

**Styling** (`styles.py`):
- CSS styling for consistent UI
- Theme configuration

## Output

The framework generates:
- **Training Results**: Metrics and performance on training data
- **Testing Results**: Out-of-sample performance metrics
- **Forecasts**: Point predictions and confidence intervals
- **Model Files**: Serialized trained models for reuse
- **Analysis Reports**: Detailed results with comparisons

Results are exported to the `output/` directory with:
- CSV files containing metrics and forecasts
- Summary statistics and rankings
- Model configuration details

## Notes

- Models are trained with a default testing horizon of 52 periods
- Hyperparameter optimization uses randomized search
- Trained models are cached for reuse (configurable via `LOAD_MODEL` flag)
- Data preprocessors maintain scaling parameters for consistent transformations

### Model Parameters

Models support automatic hyperparameter tuning with customizable parameter grids. examples below:

- **ARIMA**: Order (p,d,q), seasonal components, trend terms
- **Prophet**: Changepoint sensitivity, seasonality modes, holiday effects
- **Gradient Boosting**: Number of lags, tree parameters, regularization

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **MAPE**: Mean Absolute Percentage Error
- **SMAPE**: Symmetric Mean Absolute Percentage Error
- **Weighted MAPE**: Value-weighted error calculation
- **Cumulative Error Percentage**: Total forecast bias
- **Error Standard Deviation**: Forecast uncertainty measure

## Output

Results are automatically exported to:

- `output/training_results.csv`: In-sample performance metrics
- `output/testing_results.csv`: Out-of-sample forecast accuracy
- `output/fits/`: Model fitted values by category
- `output/forecasts/`: Future predictions by category

## Example Datasets

The framework has been tested with various time series including:

- TSA checkpoint travel counts
- Walmart sales data
- Weather data (temperature, precipitation)
- Economic indicators (gold prices, gas prices)
- Health data (COVID cases, influenza)

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- statsmodels, prophet
- PyMC, ArviZ (for Bayesian models)
- LightGBM (for gradient boosting)

## License

This project is available for research and educational purposes.