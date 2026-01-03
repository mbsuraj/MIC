import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import os

class BayesianForecaster(Forecaster):
    def __init__(self, data, data_freq='W-MON', name="bayesian_model"):
        super().__init__()
        self.baseline = None
        self.trace = None
        self.data = data
        self.trend = None
        self.seasonal = None
        self.seasonal_periods = self._get_seasonal_periods(data_freq)
        self.model = None
        self.metrics = {}
        self.name = name
        self.path = self.get_cache_path(name)
        self.fitted_values = None
        self.forecast_values = None

    def _get_seasonal_periods(self, data_freq):
        output_dict = {"Day": 365, "MonthBegin": 12, "Week": 52, 'W-MON': 52}
        return output_dict.get(data_freq, 52)

    def fit(self, params):
        """
        Fit the Bayesian model with given parameters.
        """
        # For Bayesian models, params would contain hyperparameters
        # This is a simplified version - actual implementation would depend on specific parameters
        values = self.data.values.flatten()
        with pm.Model() as model:
            baseline = pm.Normal('baseline', mu=np.median(values), sigma=params.get('baseline_sigma', 2))
            trend = pm.Normal('trend', mu=0, sigma=params.get('trend_sigma', 1))
            seasonal_raw = pm.Normal('seasonal_raw', mu=0, sigma=params.get('seasonal_sigma', 2), shape=self.seasonal_periods)
            sigma = pm.HalfNormal('sigma', sigma=params.get('obs_sigma', 3))
            seasonal = pm.Deterministic('seasonal', seasonal_raw - pm.math.mean(seasonal_raw))
            
            t = np.arange(self.data.shape[0])
            seasonal_idx = t % self.seasonal_periods
            mean = baseline + trend * t + seasonal[seasonal_idx]
            obs = pm.Normal('obs', mu=mean, sigma=sigma, observed=self.data.values)
            
            self.trace = pm.sample(draws=params.get('draws', 1000), tune=params.get('tune', 500), return_inferencedata=False, chains=2)
        
        self.baseline = np.median(self.trace['baseline'])
        self.trend = np.median(self.trace['trend'])
        self.seasonal = np.median(self.trace['seasonal'], axis=0)
        self.fitted_values = self._get_fitted_values()

    def search_and_fit(self):
        values = self.data.values.flatten()
        with pm.Model() as model:
            baseline_mu = np.median(values)

            # Hyper-priors for variance parameters
            baseline_sigma_prior = pm.HalfNormal('baseline_sigma_prior', sigma=2)
            trend_sigma_prior = pm.HalfNormal('trend_sigma_prior', sigma=1)
            seasonal_sigma_prior = pm.HalfNormal('seasonal_sigma_prior', sigma=2)
            obs_sigma_prior = pm.HalfNormal('obs_sigma_prior', sigma=3)

            # Then use these in your existing priors
            baseline = pm.Normal('baseline', mu=baseline_mu, sigma=baseline_sigma_prior)
            trend = pm.Normal('trend', mu=0, sigma=trend_sigma_prior)
            seasonal_raw = pm.Normal('seasonal_raw', mu=0, sigma=seasonal_sigma_prior, shape=self.seasonal_periods)
            sigma = pm.HalfNormal('sigma', sigma=obs_sigma_prior)
            seasonal = pm.Deterministic('seasonal', seasonal_raw - pm.math.mean(seasonal_raw))  # Zero-sum constraint

            # Time index and seasonal index
            t = np.arange(self.data.shape[0])
            seasonal_idx = t % self.seasonal_periods

            # Mean with trend and seasonality
            mean = baseline + trend * t + seasonal[seasonal_idx]

            # Likelihood
            obs = pm.Normal('obs', mu=mean, sigma=sigma, observed=self.data.values)

            # Sample
            self.trace = pm.sample(draws=10_500,
                                   tune=1_000,
                                   return_inferencedata=False,
                                   chains=4,
                                   nuts_sampler='pymc',
                                   )

        self.baseline = np.median(self.trace['baseline'])
        self.trend = np.median(self.trace['trend'])
        self.seasonal = np.median(self.trace['seasonal'], axis=0)

        self.fitted_values = self._get_fitted_values()

    def _get_fitted_values(self):
        if self.trace is None:
            raise ValueError("The model has not been fitted yet.")
        t = np.arange(self.data.shape[0])
        seasonal_idx = t % self.seasonal_periods
        return pd.DataFrame({"value": self.baseline + self.trend * t + self.seasonal[seasonal_idx]}, index=self.data.index)

    def forecast(self, steps):
        if self.trace is None:
            raise ValueError("Model not fitted.")
        start = self.data.shape[0]
        t = np.arange(start, start + steps)
        seasonal_idx = t % self.seasonal_periods
        forecasted_values = self.baseline + self.trend * t + self.seasonal[seasonal_idx]
        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq="W-MON")[1:]
        return pd.Series(forecasted_values, index=forecast_index)

    def plot_fit_vs_actual(self, steps):
        if self.trace is None:
            raise ValueError("The model has not been fitted yet.")

        fitted_values = self._get_fitted_values()
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
        plt.savefig(f"../plots/{self.name}_fit_forecast_plot.png")

    def save_model(self, path=None):
        if self.trace is None:
            raise ValueError("Model not fitted.")
        if path is not None:
            self.path = path

        dict_to_save = {
            'baseline': self.baseline,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'trace': self.trace,
            'seasonal_periods': self.seasonal_periods,
            'fitted_values': self.fitted_values,
            'data': self.data,
            'name': self.name,
            # 'path': self.path
        }
        with open(self.path, 'wb') as f:
            cloudpickle.dump(dict_to_save, f)
        print(f"Model saved to {self.path}")

    def load_model(self):
        try:
            with open(self.path, 'rb') as f:
                model_dict = cloudpickle.load(f)
            self.trace = model_dict['trace']
            self.baseline = model_dict['baseline']
            self.trend = model_dict['trend']
            self.seasonal = model_dict['seasonal']
            self.seasonal_periods = model_dict['seasonal_periods']
            self.fitted_values = model_dict['fitted_values']
            self.data = model_dict['data']
            self.name = model_dict['name']
            # self.path = model_dict['path']
            print(f"Model loaded from {self.path}")
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.search_and_fit()

    def output(self):
        if self.trace is not None:
            model_params = {
                'baseline': self.baseline,
                'trend': self.trend,
                'seasonal': self.seasonal.tolist(),
            }
            print(f"Model Parameters: {model_params}")
        else:
            print("Model is not fitted yet.")

    def log_metrics(self):
        """
        Log the metrics collected during training (or testing in future extensions).
        """
        for metric, value in self.metrics.items():
            print(f"{metric}: {value}")
