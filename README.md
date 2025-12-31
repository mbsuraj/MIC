# MIC (Make It Certain)

A comprehensive time series forecasting framework that implements multiple parametric and non-parametric forecasting models with automated hyperparameter tuning and performance evaluation.

## Overview

MIC is designed to provide certainty in time series predictions by comparing multiple forecasting approaches across various datasets. The framework automatically handles data preprocessing, model training, hyperparameter optimization, and performance evaluation.

## Features

- **Multiple Forecasting Models**: Supports both parametric and non-parametric approaches
- **Automated Data Processing**: Handles data loading, cleaning, and preprocessing with scaling
- **Hyperparameter Optimization**: Randomized search for optimal model parameters
- **Performance Evaluation**: Comprehensive metrics including MAPE, SMAPE, weighted MAPE, and error statistics
- **Model Persistence**: Save and load trained models for reuse
- **Batch Processing**: Process multiple datasets simultaneously
- **Results Export**: Detailed training and testing results with forecasts

## Supported Models

### Parametric Models
- **ARIMA**: AutoRegressive Integrated Moving Average with seasonal components
- **ETS**: Exponential Smoothing State Space models
- **Bayesian Forecaster**: Bayesian time series modeling
- **Bayesian SSM**: Bayesian State Space Models

### Non-Parametric Models
- **Gradient Boosting**: LightGBM-based forecasting with lag features
- **Prophet**: Facebook's Prophet for time series with trend and seasonality

## Project Structure

```
MIC/
├── src/
│   ├── common/
│   │   ├── experimenter.py      # Main experiment orchestrator
│   │   ├── dataGenerator.py     # Data loading and preprocessing
│   │   ├── dataPreprocessor.py  # Data scaling and transformation
│   │   ├── forecaster.py        # Abstract base class for models
│   │   └── utils.py            # Evaluation metrics
│   └── models/                 # Individual forecasting models
├── config/
│   └── data_config.json        # Dataset configuration
├── data/                       # Input datasets
├── cache/                      # Saved models
├── output/                     # Results and forecasts
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from src.common.experimenter import Experimenter

# Run complete experiment pipeline
experimenter = Experimenter()
experimenter.run_experiment()
```

### Custom Workflow

```python
# Load and preprocess data
experimenter = Experimenter()
experimenter.load_datasets()
experimenter.split_train_test()

# Train specific models
experimenter.load_forecasters()
experimenter.define_fit_and_save_models()

# Generate forecasts
experimenter.forecast_from_models()
experimenter.export_experiment_results()
```

## Configuration

### Data Configuration

Configure datasets in `config/data_config.json`:

```json
{
  "dataset_name": {
    "format": "%Y-%m-%d",
    "freq": "W-MON"
  }
}
```

### Model Parameters

Models support automatic hyperparameter tuning with customizable parameter grids. Key parameters include:

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

## Advanced Features

### Model Persistence

```python
# Models are automatically saved during training
model.save_model()

# Load previously trained models
model.load_model()
```

### Custom Metrics

Add custom evaluation metrics in `src/common/utils.py`:

```python
def custom_metric(y_true, y_pred):
    # Your metric implementation
    return metric_value
```

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- statsmodels, prophet
- PyMC, ArviZ (for Bayesian models)
- LightGBM (for gradient boosting)

## License

This project is available for research and educational purposes.