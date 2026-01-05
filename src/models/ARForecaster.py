import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os
import random

class ARForecaster(Forecaster):
    def __init__(self, data, lags=1, name="ar_model", data_freq='W-Mon'):
        """
        Initialize the ARForecaster with time series data and number of lags.

        Parameters:
        data (array-like): The time series data to be used for forecasting.
        lags (int): The number of lags to use in the autoregressive model.
        """
        super().__init__()
        self.data = data
        self.data_freq = data_freq
        self.lags = lags
        self.model = None
        self.fitted_model = None
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.fitted_values = None
        self.forecast_values = None

    def perform_random_search(self, param_grid, n_iter=50):
        """
        Custom random search for statsmodels AutoReg.

        Args:
            param_grid: Dictionary of parameters to search
            n_iter: Number of iterations for random search

        Returns:
            Best model and its parameters
        """
        best_rmse = float('inf')
        best_model = None
        best_params = None
        all_metrics = []

        # Generate random combinations
        param_combinations = []
        param_keys = list(param_grid.keys())

        for _ in range(n_iter):
            combination = {
                key: random.choice(param_grid[key])
                for key in param_keys
            }
            param_combinations.append(combination)

        # Try each combination
        for params in param_combinations:
            try:
                # Fit model with current parameters
                model = AutoReg(
                    self.data,
                    lags=params['lags'],
                    trend=params['trend'],
                    seasonal=params['seasonal']
                )
                fitted_model = model.fit()

                # Calculate RMSE
                predictions = fitted_model.predict(start=params['lags'], end=len(self.data) - 1)
                actual_values = self.data[params['lags']:]
                rmse = np.sqrt(np.mean((predictions.values - actual_values.values.flatten()) ** 2))

                # Store metrics for this iteration
                metrics = {
                    'params': params,
                    'rmse': rmse,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                all_metrics.append(metrics)

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = fitted_model
                    best_params = params

            except Exception as e:
                print(f"Skipping parameters {params} due to error: {str(e)}")
                continue

        # Store results in the same format as before
        self.random_search_results = {
            'best_params': best_params,
            'best_score': best_rmse,
            'cv_results': all_metrics
        }
        self.lags = best_params['lags']
        return best_model

    def fit(self, params):
        """
        Fit the AR model with given parameters.
        """
        model = AutoReg(
            self.data,
            lags=params['lags'],
            trend=params['trend'],
            seasonal=params['seasonal']
        )
        self.fitted_model = model.fit()
        self.fitted_values = self.fitted_model.predict(start=params['lags'], end=len(self.data) - 1)
        self.lags = params['lags']

    def search_and_fit(self):
        """
        Fit the AR model on the provided data with parameter search.
        """
        # Define parameter grid
        param_grid = {
            'lags': list(range(1, self.lags + 1)),
            'trend': ['n', 'c', 't', 'ct'],
            'seasonal': [True, False]
        }

        # Perform random search
        self.fitted_model = self.perform_random_search(param_grid)

        # Get predictions
        self.fitted_values = self.fitted_model.predict(start=self.lags, end=len(self.data) - 1)
        self.save_search_results('ar_forecaster')

    def log_metrics(self):
        """
        Log the metrics collected during training (or testing in future extensions).
        """
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")

    def save_model(self, path=None):
        """
        Save the fitted AR model to a file using pickle.

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
        Load a previously saved AR model from a file using pickle.

        Parameters:
        path (str): The file path from which the model should be loaded.
        """
        try:
            with open(self.path, 'rb') as f:
                self.fitted_model = pickle.load(f)
            print(f"Model loaded from {self.path}")
            self.fitted_values = self.fitted_model.predict(start=self.lags, end=len(self.data) - 1)
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.search_and_fit()


    def output(self):
        """
        Output the model's coefficients or other relevant information.
        """
        if self.fitted_model is not None:
            print(f"Model Coefficients: {self.fitted_model.params}")
        else:
            print("Model is not fitted yet.")

    def forecast(self, steps):
        """
        Forecast future values using the trained AR model.

        Parameters:
        steps (int): The number of future steps to forecast.

        Returns:
        pd.Series: Forecasted values with a DateTimeIndex continuing from the end of the training data.
        """
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        start = self.data.index[-1]
        end = start + timedelta(weeks=steps)
        # Generate forecasted values
        forecasted_values = self.fitted_model.predict(start=start, end=end,
                                                      dynamic=False)

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
        fitted_values = self.fitted_model.predict(start=self.lags, end=len(self.data) - 1)

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

        mape = round(mean_absolute_percentage_error(self.data.values[self.lags:], fitted_values), 2)
        # Add titles and labels
        plt.title(f'{self.name}: Actual-Fit-Forecasts; Training MAPE: {mape}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{os.getcwd()}/plots/{self.name}_fit_forecast_plot.png")

# Example usage:
# ar_forecaster = ARForecaster(data=data_series, lags=5)
# ar_forecaster.fit()
# ar_forecaster.save_model('ar_model.pkl')
# ar_forecaster.plot_fit_vs_actual(steps=10)

# To load the model later:
# ar_forecaster.load_model('ar_model.pkl')
# forecasted_values = ar_forecaster.forecast(steps=10)
# ar_forecaster.plot_fit_vs_actual(steps=10)