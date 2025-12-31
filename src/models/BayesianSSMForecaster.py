import cloudpickle
import numpy as np
import pandas as pd
import pymc as pm
from src.pymc_statespace.models import structural as st
from src.common.forecaster import Forecaster
import pytensor.tensor as pt
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import os

class BayesianSSMForecaster(Forecaster):
    def __init__(self, data, data_freq='W-MON', name="bayesian_ssm_model"):
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

    def _build_ssm(self):
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        season = st.FrequencySeasonality(season_length=self.seasonal_periods, name='season', innovations=True)
        # season = st.TimeSeasonality(season_length=self.seasonal_periods, name='season', innovations=False)
        return (grw + season).build()

    def fit(self):
        values = pd.DataFrame(self.data.values.flatten(), index=self.data.index)
        self.model = self._build_ssm()
        with pm.Model(coords=self.model.coords):
            P0 = pm.Deterministic('P0', pt.eye(self.model.k_states) * 1.0, dims=self.model.param_dims['P0'])
            initial_trend = pm.Deterministic('initial_trend', pt.zeros(1), dims=self.model.param_dims['initial_trend'])
            # season_coefs = pm.Normal('season_coefs', sigma=1e-2, dims=self.model.param_dims['season_coefs'])
            season = pm.Normal('season', sigma=0.01, dims=self.model.param_dims['season'])
            sigma_season = pm.HalfNormal('sigma_season', sigma=0.01)
            sigma_trend = pm.HalfNormal('sigma_trend', sigma=0.01, dims=self.model.param_dims['sigma_trend'])
            self.model.build_statespace_graph(values)
            self.trace = pm.sample(draws=500,
                                   tune=100,
                                   chains=4,
                                   nuts_sampler='pymc',
                                   return_inferencedata=False,
                                   target_accept=0.9,
                                   )

        self.fitted_values = self._get_fitted_values()

    def _get_fitted_values(self):
        return pd.DataFrame({
            "value": self._forecast_values(self.data.index[0], self.data.shape[0])
        }, index=self.data.index)

    def _forecast_values(self, start, periods):
        fc = self.model.forecast(idata=self.trace, start=start, periods=periods)
        vals = fc.forecast_observed.mean(dim=['chain', 'draw']).values.flatten()
        return vals

    def forecast(self, steps):
        forecasted_values = self._forecast_values(self.data.index[-1], steps).flatten()
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
            'model': self.model,
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
            self.model = model_dict['model']
            self.trace = model_dict['trace']
            self.seasonal_periods = model_dict['seasonal_periods']
            self.fitted_values = model_dict['fitted_values']
            self.data = model_dict['data']
            self.name = model_dict['name']
            # self.path = model_dict['path']
            print(f"Model loaded from {self.path}")
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.fit()

    def output(self):
        if self.trace is not None:
            model_params = {
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
