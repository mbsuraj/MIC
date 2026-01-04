import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os
import random

class ETSForecaster(Forecaster):
    def __init__(self, data, error='add', trend='add', seasonal='add', seasonal_periods=52, data_freq='W-MON', name="ETS_model", data_freq_type='week'):
        super().__init__()
        self.data = data
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = self._get_seasonal_periods(data_freq_type)
        self.model = None
        self.fitted_model = None
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.fitted_values = None
        self.forecast_values = None
        self.data_freq_type = data_freq_type
        self.data_freq = data_freq

    def _get_seasonal_periods(self, data_freq_type):
        output_dict = {"day": [7, 30, 120, 365],
                       "week": [4, 12, 52],
                       "month": [3, 12],
                       'year': [1]}
        return output_dict[data_freq_type]

    def perform_random_search(self, param_grid, n_iter=50):
        best_rmse = float('inf')
        best_model = None
        best_params = None
        all_metrics = []
        valid_combinations_found = 0
        max_attempts = n_iter * 3
        attempts = 0

        while valid_combinations_found < n_iter and attempts < max_attempts:
            attempts += 1
            params = {key: random.choice(param_grid[key]) for key in param_grid}

            try:
                model = ETSModel(
                    self.data.iloc[:, 0] if not isinstance(self.data, pd.Series) else self.data,
                    error=params['error'],
                    trend=params['trend'],
                    seasonal=params['seasonal'],
                    seasonal_periods=params['seasonal_periods'],
                    damped_trend=params['damped_trend']
                )
                fitted_model = model.fit()

                predictions = fitted_model.fittedvalues
                rmse = np.sqrt(np.mean((predictions.values - self.data.values.flatten()) ** 2))

                metrics = {
                    'params': params,
                    'rmse': rmse,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                all_metrics.append(metrics)
                valid_combinations_found += 1

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = fitted_model
                    best_params = params

            except Exception as e:
                print(f"Skipping parameters {params} due to error: {str(e)}")
                continue

        self.random_search_results = {
            'best_params': best_params,
            'best_score': best_rmse,
            'cv_results': all_metrics
        }

        return best_model

    def fit(self, params):
        """
        Fit the ETS model with given parameters.
        """
        model = ETSModel(
            self.data.iloc[:, 0] if not isinstance(self.data, pd.Series) else self.data,
            error=params['error'],
            trend=params['trend'],
            seasonal=params['seasonal'],
            seasonal_periods=params['seasonal_periods'],
            damped_trend=params['damped_trend']
        )
        self.fitted_model = model.fit()
        self.fitted_values = self.fitted_model.fittedvalues

    def search_and_fit(self):
        param_grid = {
            'error': ['add', 'mul'],
            'trend': [None, 'add', 'mul'],
            'seasonal': [None, 'add', 'mul'],
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': [True, False]
        }

        self.fitted_model = self.perform_random_search(param_grid)
        self.fitted_values = self.fitted_model.fittedvalues
        self.save_search_results('ets_forecaster')

    def forecast(self, steps):
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")

        forecasted_values = self.fitted_model.forecast(steps=steps)
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self.data_freq)[1:]
        return pd.Series(forecasted_values, index=forecast_index)
    
    def forecast_with_intervals(self, steps, alpha=0.05):
        """
        Forecast with confidence intervals.
        
        Parameters:
        steps (int): Number of steps to forecast
        alpha (float): Significance level (0.05 for 95% confidence)
        
        Returns:
        dict: {'forecast': Series, 'lower': Series, 'upper': Series}
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
            
        forecast_result = self.fitted_model.get_prediction(start=self.data.index[-1],
                                                           end=self.data.index[-1] + pd.Timedelta(days=steps * 7),
                                                           simulate=True,
                                                           simulate_repetitions=3000)
        conf_int = forecast_result.pred_int(alpha=alpha)
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self.data_freq)[1:]

        return {
            'forecast': pd.Series(forecast_result.predicted_mean, index=forecast_index),
            'lower': pd.Series(conf_int.iloc[:, 0], index=forecast_index),
            'upper': pd.Series(conf_int.iloc[:, 1], index=forecast_index)
        }

    def plot_fit_vs_actual(self, steps):
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        fitted_values = self.fitted_model.fittedvalues
        forecasted_values = self.forecast(steps)

        plt.figure(figsize=(10, 6))
        plt.plot(self.data.iloc[-50:], color='black', label='Actual')
        plt.plot(self.data.index[-50:], fitted_values[-50:], 'y--', label='Fitted')
        plt.plot(forecasted_values, 'r--', label='Forecasted')

        mape = round(mean_absolute_percentage_error(self.data.values, fitted_values), 2)
        plt.title(f'{self.name}: Actual-Fit-Forecasts; Training MAPE: {mape}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{os.getcwd()}/plots/{self.name}_fit_forecast_plot.png")

    def save_model(self, path=None):
        if self.fitted_model is None:
            raise ValueError("Model not fitted.")
        if path is not None:
            self.path = path
        with open(self.path, 'wb') as f:
            pickle.dump(self.fitted_model, f)
        print(f"Model saved to {self.path}")

    def load_model(self):
        try:
            with open(self.path, 'rb') as f:
                self.fitted_model = pickle.load(f)
            print(f"Model loaded from {self.path}")
            self.fitted_values = self.fitted_model.fittedvalues
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.search_and_fit()

    def output(self):
        if self.fitted_model is not None:
            print(f"Model Parameters: {self.fitted_model.model.params}")
        else:
            print("Model is not fitted yet.")

    def log_metrics(self):
        """
        Log the metrics collected during training (or testing in future extensions).
        """
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")
