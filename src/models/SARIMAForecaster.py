import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

class SARIMAForecaster(Forecaster):
    def __init__(self, data, order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), name="sarima_model", data_freq='W-Mon'):
        """
        Initialize the SARIMAForecaster with time series data and model order.

        Parameters:
        data (array-like): The time series data to be used for forecasting.
        order (tuple): The (p, d, q) order of the ARIMA model.
        seasonal_order (tuple): The (P, D, Q, s) seasonal order of the SARIMA model.
        name (str): The name for the model, used in paths for saving/loading.
        """
        super().__init__()
        self.data = data
        self.data_freq = data_freq
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)

    def fit(self, params):
        """
        Fit the SARIMA model with given parameters.
        """
        order = (params['p'], params['d'], params['q'])
        seasonal_order = (params['P'], params['D'], params['Q'], params['s']) if params['seasonal'] else None
        self.model = SARIMAX(self.data, order=order, seasonal_order=seasonal_order)
        self.fitted_model = self.model.fit(disp=False)

    def search_and_fit(self):
        """
        Fit the SARIMA model on the provided data.
        """
        self.model = SARIMAX(self.data, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        print("Model fitted successfully.")

    def log_metrics(self):
        """
        Log the metrics collected during training (or testing in future extensions).
        """
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")

    def save_model(self, path=None):
        """
        Save the fitted SARIMA model to a file using pickle.

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
        Load a previously saved SARIMA model from a file using pickle.

        Parameters:
        path (str): The file path from which the model should be loaded.
        """
        try:
            with open(self.path, 'rb') as f:
                self.fitted_model = pickle.load(f)
            print(f"Model loaded from {self.path}")
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.search_and_fit()

    def output(self):
        """
        Output the model's summary or other relevant information.
        """
        if self.fitted_model is not None:
            print(self.fitted_model.summary())
        else:
            print("Model is not fitted yet.")

    def forecast(self, steps):
        """
        Forecast future values using the trained SARIMA model.

        Parameters:
        steps (int): The number of future steps to forecast.

        Returns:
        pd.Series: Forecasted values with a DateTimeIndex continuing from the end of the training data.
        """
        if self.fitted_model is None:
            raise ValueError("The model has not been fitted yet.")

        forecasted_values = self.fitted_model.get_forecast(steps=steps).predicted_mean

        # Create a datetime index for the forecast
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq=self.data_freq)[1:]

        # Return as a pandas Series with DateTimeIndex
        return pd.Series(forecasted_values, index=forecast_index)

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
        plt.savefig(f"../plots/{self.name}_fit_forecast_plot.png")

# Example usage:
# sarima_forecaster = SARIMAForecaster(data=data_series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
# sarima_forecaster.fit()
# sarima_forecaster.save_model('sarima_model.pkl')
# sarima_forecaster.plot_fit_vs_actual(steps=10)

# To load the model later:
# sarima_forecaster.load_model('sarima_model.pkl')
# forecasted_values = sarima_forecaster.forecast(steps=10)
# sarima_forecaster.plot_fit_vs_actual(steps=10)
