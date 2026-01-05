import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os
import random

class HoltWintersForecaster(Forecaster):
    def __init__(self, data, trend='add', seasonal='add', seasonal_periods=365, data_freq='W-Mon', name="HoltWinters_model", data_freq_type='week'):
        """
        Initialize the HoltWintersForecaster with time series data and model configuration.

        Parameters:
        data (array-like): The time series data to be used for forecasting.
        trend (str or None): The type of trend component ('add', 'mul', or None).
        seasonal (str or None): The type of seasonal component ('add', 'mul', or None).
        seasonal_periods (int): The number of periods in a season (e.g., 12 for monthly data with annual seasonality).
        """
        super().__init__()
        self.data = data
        self.data_freq = data_freq
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = self._get_seasonal_periods(data_freq_type)
        # self.data_freq = data_freq
        self.model = None
        self.fitted_model = None
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.fitted_values = None
        self.forecast_values = None
        self.data_freq_type = data_freq_type

    def _get_seasonal_periods(self, data_freq_type):
        output_dict = {"day": [7, 30, 120, 365],
                       "week": [4, 12, 52],
                       "month": [3, 12],
                       'year': [1]}
        return output_dict[data_freq_type]

    def perform_random_search(self, param_grid, n_iter=50):
        """
        Custom random search for statsmodels Holt-Winters ExponentialSmoothing.
        Continues searching until n_iter valid parameter combinations are found.

        Args:
            param_grid: Dictionary of parameters to search
            n_iter: Number of valid parameter combinations to find

        Returns:
            Best model and its parameters
        """
        best_rmse = float('inf')
        best_model = None
        best_params = None
        all_metrics = []
        valid_combinations_found = 0
        max_attempts = n_iter * 3  # Limit total attempts to prevent infinite loops
        attempts = 0

        while valid_combinations_found < n_iter and attempts < max_attempts:
            attempts += 1
            # Generate random combination
            params = {
                key: random.choice(param_grid[key])
                for key in param_grid.keys()
            }

            try:
                # Fit model with current parameters
                model = ExponentialSmoothing(
                    self.data,
                    trend=params['trend'],
                    damped_trend=params['damped_trend'],
                    seasonal=params['seasonal'],
                    seasonal_periods=params['seasonal_periods'] if params['seasonal'] is not None else None,
                    initialization_method=params['initialization_method'],
                )
                fitted_model = model.fit()

                # Calculate RMSE
                predictions = fitted_model.fittedvalues
                rmse = np.sqrt(np.mean((predictions.values - self.data.values.flatten()) ** 2))

                # Store metrics for this iteration
                metrics = {
                    'params': params,
                    'rmse': rmse,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'aicc': fitted_model.aicc
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

        if valid_combinations_found < n_iter:
            print(f"Warning: Only found {valid_combinations_found} valid parameter combinations "
                  f"out of {n_iter} requested after {attempts} attempts.")
        else:
            print(f"Successfully found {valid_combinations_found} valid parameter combinations "
                  f"after {attempts} attempts.")

        # Store results in the same format as before
        self.random_search_results = {
            'best_params': best_params,
            'best_score': best_rmse,
            'cv_results': all_metrics
        }

        return best_model

    def fit(self, params):
        """
        Fit the Holt-Winters model with given parameters.
        """
        model = ExponentialSmoothing(
            self.data,
            trend=params['trend'],
            damped_trend=params['damped_trend'],
            seasonal=params['seasonal'],
            seasonal_periods=params['seasonal_periods'] if params['seasonal'] is not None else None,
            initialization_method=params['initialization_method']
        )
        self.fitted_model = model.fit()
        self.fitted_values = self.fitted_model.fittedvalues

    def search_and_fit(self):
        """
        Fit the Holt-Winters model on the provided data with parameter search.
        """
        # Define parameter grid specific to Holt-Winters ExponentialSmoothing
        param_grid = {
            'trend': [None, 'add', 'mul'],  # Type of trend component
            'damped_trend': [True, False],  # Whether to use damped trend
            'seasonal': [None, 'add', 'mul'],  # Type of seasonal component
            'seasonal_periods': self.seasonal_periods,  # Number of periods in a complete seasonal cycle
            'initialization_method': ['estimated', 'heuristic', 'legacy-heuristic'],
        }

        # Perform random search
        self.fitted_model = self.perform_random_search(param_grid)

        # Get predictions
        self.fitted_values = self.fitted_model.fittedvalues

        # Save the search results
        self.save_search_results('holt_winters_forecaster')

    def log_metrics(self):
        """
        Log the metrics collected during training (or testing in future extensions).
        """
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")

    def save_model(self, path=None):
        """
        Save the fitted HoltWinters model to a file using pickle.

        Parameters:
        path (str): The file path where the model should be saved.
        """
        if self.fitted_model is None:
            raise ValueError("Model is not fitted yet.")

        if path is not None:
            self.path = path
        with open(self.path, 'wb') as f:
            pickle.dump(self.fitted_model, f)
        print(f"Model saved to {self.path}")

    def load_model(self):
        """
        Load a previously saved HoltWinters model from a file using pickle.

        Parameters:
        path (str): The file path from which the model should be loaded.
        """
        try:
            with open(self.path, 'rb') as f:
                self.fitted_model = pickle.load(f)
            print(f"Model loaded from {self.path}")
            self.fitted_values = self.fitted_model.fittedvalues
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.search_and_fit()

    def output(self):
        """
        Output the model's parameters or other relevant information.
        """
        if self.fitted_model is not None:
            print(f"Model Parameters: {self.fitted_model.params}")
        else:
            print("Model is not fitted yet.")

    def forecast(self, steps):
        """
        Forecast future values using the trained HoltWinters model.

        Parameters:
        steps (int): The number of future steps to forecast.

        Returns:
        pd.Series: Forecasted values with a DateTimeIndex continuing from the end of the training data.
        """
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        forecasted_values = self.fitted_model.forecast(steps=steps)

        # Create a datetime index for the forecast
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self.data_freq)[1:]

        # Return as a pandas Series with DateTimeIndex
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
            raise ValueError("The model has not been fitted yet.")

        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self.data_freq)[1:]

        conf_int = forecast_result.conf_int(alpha=alpha)

        return {
            'forecast': pd.Series(forecast_result.predicted_mean, index=forecast_index),
            'lower': pd.Series(conf_int.iloc[:, 0], index=forecast_index),
            'upper': pd.Series(conf_int.iloc[:, 1], index=forecast_index)
        }

    def plot_fit_vs_actual(self, steps):
        """
        Plot the actual data, fitted values, and forecasted values.

        - Actual data: solid black line
        - Fitted values: dotted yellow line
        - Forecasted values: dotted red line

        Parameters:
        steps (int): The number of future steps to forecast.
        """
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        # Fitted values (in-sample predictions)
        fitted_values = self.fitted_model.fittedvalues

        # Forecasted values (out-of-sample predictions)
        forecasted_values = self.forecast(steps)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot actual data (solid black line)
        plt.plot(self.data.iloc[-50:], color='black', label='Actual')

        # Plot fitted values (dotted yellow line)
        plt.plot(self.data.index[-50:], fitted_values[-50:], 'y--', label='Fitted')

        # Plot forecasted values (dotted red line)
        plt.plot(forecasted_values, 'r--', label='Forecasted')

        mape = round(mean_absolute_percentage_error(self.data.values, fitted_values), 2)
        # Add titles and labels
        plt.title(f'{self.name}: Actual-Fit-Forecasts; Training MAPE: {mape}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{os.getcwd()}/plots/{self.name}_fit_forecast_plot.png")

# Example usage:
# HoltWinters_forecaster = HoltWintersForecaster(data=data_series, trend='add', seasonal='mul', seasonal_periods=12)
# HoltWinters_forecaster.fit()
# HoltWinters_forecaster.save_model('HoltWinters_model.pkl')
# HoltWinters_forecaster.plot_fit_vs_actual(steps=10)

# To load the model later:
# HoltWinters_forecaster.load_model('HoltWinters_model.pkl')
# forecasted_values = HoltWinters_forecaster.forecast(steps=10)
# HoltWinters_forecaster.plot_fit_vs_actual(steps=10)
