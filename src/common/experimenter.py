import pandas as pd
from src.common.dataGenerator import DataGenerator
from src.common.dataPreprocessor import DataPreprocessor
from src.models.ARIMAForecaster import ARIMAForecaster
from src.models.BayesianSSMForecaster import BayesianSSMForecaster
from src.models.GBForecaster import GBForecaster
from src.models.ProphetForecaster import ProphetForecaster
from src.models.BayesianForecaster import BayesianForecaster
from src.models.ETSForecaster import ETSForecaster
import os
from src.common.utils import *

# Get absolute paths based on the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

DIRECTORY_PATH = os.path.join(PROJECT_ROOT, "data")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "data_config.json")
FITS_FORECASTS_EXPORT = os.path.join(PROJECT_ROOT, "output")
TESTING_HORIZON = 52
LOAD_MODEL = True

class Experimenter:
    def __init__(self):
        self._full_data_dict = dict()
        self.dg = DataGenerator(directory=DIRECTORY_PATH, config_path=CONFIG_PATH)
        self.all_data_dict = None
        self.testing_horizon = TESTING_HORIZON
        self._load_model = LOAD_MODEL
        self.parametric_forecasters = None
        self.nonparametric_forecasters = None
        self.testing_datasets = dict()
        self.training_datasets = dict()
        self._data = None
        self._fits_and_forecasts = None
        self.parametric_forecasters_trained = dict()
        self.nonparametric_forecasters_trained = dict()
        self.training_results = pd.DataFrame(columns=["dataset_name", "model_name"])
        self.testing_results = pd.DataFrame(columns=["dataset_name", "model_name"])
        self.preprocessors = dict()  # Store preprocessors for each dataset

    def load_parametric_forecasters(self):
        self.parametric_forecasters = {
            # "ar_forecaster": ARForecaster,
            # "holt_winters_forecaster": HoltWintersForecaster,
            "ets_forecaster": ETSForecaster,
            "arima_forecaster": ARIMAForecaster,
            # "bayesian_forecaster": BayesianForecaster,
            # "bayesian_ssm_forecaster": BayesianSSMForecaster
        }

    def _get_parametric_forecasters_kwargs(self, model_name, dataset_name):
        parametric_forecasters_kwargs = {"ar_forecaster": {"data": self._data, "lags": 5,
                                                           "name": f"{dataset_name}_ar_model"},
                                         "holt_winters_forecaster": {"data": self._data,
                                                            "data_freq": self._data.index.freq.freqstr,
                                                            "name": f"{dataset_name}_holt_winters_model"},
                                         "ets_forecaster": {"data": self._data,
                                                                     "data_freq": self._data.index.freq.freqstr,
                                                                     "name": f"{dataset_name}_ets_model"},
                                         "arima_forecaster": {"data": self._data, "order": (2, 1, 2),
                                                              "name": f"{dataset_name}_arima_model"},
                                         "bayesian_forecaster": {"data": self._data, "data_freq": self._data.index.freq.freqstr,
                                                              "name": f"{dataset_name}_bayesian_model"},
                                         "bayesian_ssm_forecaster": {"data": self._data,
                                                                 "data_freq": self._data.index.freq.freqstr,
                                                                 "name": f"{dataset_name}_bayesian_ssm_model"}
                                         }
        return parametric_forecasters_kwargs[model_name]

    def load_nonparametric_forecasters(self):
        self.nonparametric_forecasters = {
                                          "gb_forecaster": GBForecaster,
                                          # "rt_forecaster": RTForecaster,
                                          # "sv_forecaster": SVForecaster,
                                          # "nn_forecaster": NNForecaster,
                                          "prophet_forecaster": ProphetForecaster
        }

    def _get_nonparametric_forecasters_kwargs(self, model_name, dataset_name):
        nonparametric_forecasters_kwargs = {
            "gb_forecaster": {"data": self._data, "lags": 10, "name": f"{dataset_name}_gb_model"},
            "rt_forecaster": {"data": self._data, "lags": 10, "name": f"{dataset_name}_rt_model"},
            "sv_forecaster": {"data": self._data, "lags": 10, "name": f"{dataset_name}_sv_model"},
            "nn_forecaster": {"data": self._data, "lags": 10, "name": f"{dataset_name}_nn_model"},
            "prophet_forecaster": {"data": self._data, "name": f"{dataset_name}_prophet_model"}
        }
        return nonparametric_forecasters_kwargs[model_name]

    def load_forecasters(self):
        self.load_parametric_forecasters()
        self.load_nonparametric_forecasters()

    def load_datasets(self):
        self.all_data_dict = self.dg.load_data()

    def split_train_test(self):
        assert self.all_data_dict is not None, "self.all_data_dict cannot be empty. Run self.load_datasets method before split_train_test"
        for name, data in self.all_data_dict.items():
            print(name, end="\n\n")
            # Create preprocessor for each dataset
            self.preprocessors[name] = DataPreprocessor(data)
            
            # Use scaled data for train/test split
            scaled_data = self.preprocessors[name].scaled_data
            self.training_datasets[name] = scaled_data.iloc[:-self.testing_horizon]
            self.testing_datasets[name] = scaled_data.iloc[-self.testing_horizon:]

    def get_full_data(self):
        assert self.all_data_dict is not None, "self.all_data_dict cannot be empty. Run self.load_datasets method before split_train_test"
        for name, data in self.all_data_dict.items():
            print(name, end="\n\n")
            # Create preprocessor for each dataset
            self.preprocessors[name] = DataPreprocessor(data)

            # Use scaled data for train/test split
            scaled_data = self.preprocessors[name].scaled_data
            self._full_data_dict[name] = scaled_data

    def define_fit_and_save_models(self):
        self.define_fit_and_save_parametric_models()
        self.define_fit_and_save_nonparametric_models()

    def define_fit_and_save_parametric_models(self):
        for dataset_name, data in self.training_datasets.items():
            self.parametric_forecasters_trained[dataset_name] = dict()
            if data is None:
                continue
            self._data = data.dropna()
            _fits_export = pd.DataFrame(self.preprocessors[dataset_name].inverse_transform(self._data.values), columns=['y_true'], index=self._data.index)
            for model_name, model in self.parametric_forecasters.items():
                model_kwargs = self._get_parametric_forecasters_kwargs(model_name=model_name,
                                                                       dataset_name=dataset_name)
                defined_model = model(**model_kwargs)
                if self._load_model:
                    defined_model.load_model()
                else:
                    defined_model.fit()
                y_true = self._data if model_name != "ar_forecaster" else self._data.iloc[defined_model.lags:]
                y_pred = defined_model.fitted_values
                y_true_original, y_pred_original = self._get_inverse_values(y_true, y_pred, dataset_name)
                _fits_export[model_name] = y_pred_original
                metrics = self._calculate_metrics(y_true_original, y_pred_original, dataset_name)
                self._log_data_model_training_metrics(dataset_name, model_name, metrics)
                defined_model.save_model()
                self.parametric_forecasters_trained[dataset_name][model_name] = defined_model
            _fits_export.to_csv(f"{FITS_FORECASTS_EXPORT}/fits/parametric/{dataset_name}.csv")

    def _get_inverse_values(self, y_true, y_pred, dataset_name):
        # Scale back to original units for meaningful metrics
        preprocessor = self.preprocessors[dataset_name]
        y_true_orig = preprocessor.inverse_transform(y_true.values)
        y_pred_orig = preprocessor.inverse_transform(y_pred.values)
        return y_true_orig, y_pred_orig

    def _calculate_metrics(self, y_true, y_pred, dataset_name):
        mape = round(mean_absolute_percentage_error(y_true, y_pred), 2)
        smape_value = round(smape(y_true, y_pred), 2)
        cumulative_error_percentage_value = round(cumulative_error_percentage(y_true, y_pred), 2)
        w_mape = weighted_mape(y_true, y_pred)
        std = error_std(y_true, y_pred)
        res = {
            "mape": mape,
            "smape": smape_value,
            "cumulative_error_percentage": cumulative_error_percentage_value,
            "weighted_mape": w_mape,
            "error_std": std
        }
        return res


    def define_fit_and_save_nonparametric_models(self):
        for dataset_name, data in self.training_datasets.items():
            self.nonparametric_forecasters_trained[dataset_name] = dict()
            if data is None:
                continue
            self._data = data.dropna()
            _fits_export = pd.DataFrame(self.preprocessors[dataset_name].inverse_transform(self._data.values), columns=['y_true'], index=self._data.index)
            for model_name, model in self.nonparametric_forecasters.items():
                model_kwargs = self._get_nonparametric_forecasters_kwargs(model_name=model_name,
                                                                       dataset_name=dataset_name)
                defined_model = model(**model_kwargs)
                if self._load_model:
                    defined_model.load_model()
                else:
                    defined_model.fit()
                y_true = self._data if model_name == "prophet_forecaster" else self._data.iloc[defined_model.lags:]
                y_pred = pd.Series(defined_model.fitted_values.yhat.values, index=self._data.index) if model_name == "prophet_forecaster" else pd.Series(defined_model.fitted_values, index=self._data.index[defined_model.lags:])
                y_true_original, y_pred_original = self._get_inverse_values(y_true, y_pred, dataset_name)
                _fits_export[model_name] = y_pred_original if model_name == "prophet_forecaster" else np.concatenate((np.full(defined_model.lags, np.nan), y_pred_original), axis=0)
                metrics = self._calculate_metrics(y_true_original, y_pred_original, dataset_name)
                self._log_data_model_training_metrics(dataset_name, model_name, metrics)
                defined_model.save_model()
                self.nonparametric_forecasters_trained[dataset_name][model_name] = defined_model
            _fits_export.to_csv(f"{FITS_FORECASTS_EXPORT}/fits/non_parametric/{dataset_name}.csv")

    def _log_data_model_training_metrics(self, dataset_name, model_name, metrics):
        data_dict = {"dataset_name": [dataset_name],
                     "model_name": [model_name]}
        data_dict.update(metrics)
        temp_df = pd.DataFrame(data_dict)
        self.training_results = pd.concat([self.training_results, temp_df], axis=0)

    def _log_data_model_testing_metrics(self, dataset_name, model_name, metrics):
        data_dict = {"dataset_name": [dataset_name],
                     "model_name": [model_name]}
        data_dict.update(metrics)
        temp_df = pd.DataFrame(data_dict)
        self.testing_results = pd.concat([self.testing_results, temp_df], axis=0)

    def forecast_from_models(self):
        self.forecast_from_parametric_models()
        self.forecast_from_nonparametric_models()

    def forecast_from_parametric_models(self):
        for dataset_name, data in self.testing_datasets.items():
            if data is None:
                continue
            self._data = data
            _forecasts_export = pd.DataFrame(self.preprocessors[dataset_name].inverse_transform(self._data.values), columns=['y_true'], index=self._data.index)
            for model_name, model in self.parametric_forecasters_trained[dataset_name].items():
                forecast = model.forecast(52) if model_name != "garch_forecaster" else model._inverse_scale_data(
                    model.forecast(52))
                y_true = self._data
                y_pred = pd.Series(forecast.values, index=self._data.index)
                y_true_original, y_pred_original = self._get_inverse_values(y_true, y_pred, dataset_name)
                _forecasts_export[model_name] = y_pred_original
                metrics = self._calculate_metrics(y_true_original, y_pred_original, dataset_name)
                self._log_data_model_testing_metrics(dataset_name, model_name, metrics)
            _forecasts_export.to_csv(f"{FITS_FORECASTS_EXPORT}/forecasts/parametric/{dataset_name}.csv")

    def forecast_from_nonparametric_models(self):
        for dataset_name, data in self.testing_datasets.items():
            if data is None:
                continue
            self._data = data
            _forecasts_export = pd.DataFrame(self.preprocessors[dataset_name].inverse_transform(self._data.values), columns=['y_true'], index=self._data.index)
            for model_name, model in self.nonparametric_forecasters_trained[dataset_name].items():
                forecast = model.forecast(52)
                y_true = self._data
                y_pred = pd.Series(forecast.yhat.values, index=self._data.index) if model_name == "prophet_forecaster" else pd.Series(forecast.values, index=self._data.index)
                y_true_original, y_pred_original = self._get_inverse_values(y_true, y_pred, dataset_name)
                _forecasts_export[model_name] = y_pred_original
                metrics = self._calculate_metrics(y_true_original, y_pred_original, dataset_name)
                self._log_data_model_testing_metrics(dataset_name, model_name, metrics)
            _forecasts_export.to_csv(f"{FITS_FORECASTS_EXPORT}/forecasts/non_parametric/{dataset_name}.csv")

    def export_experiment_results(self):
        self.training_results.to_csv(os.path.join(PROJECT_ROOT, "output", "training_results.csv"), index=False)
        self.testing_results.to_csv(os.path.join(PROJECT_ROOT, "output", "testing_results.csv"), index=False)

    def retrain_best_model_and_forecast_future(self, best_model_name, dataset_name, periods=52):
        """Retrain the best model on full dataset and generate future forecasts"""
        try:
            target_dataset = dataset_name
            model_type = None
            if best_model_name in self.parametric_forecasters:
                model_type = 'parametric'
            elif best_model_name in self.nonparametric_forecasters:
                model_type = 'nonparametric'
            
            # Get full dataset (training + testing)
            full_data = self._full_data_dict[target_dataset]
            self._data = full_data.dropna()
            
            # Get model class and kwargs
            if model_type == 'parametric':
                model_class = self.parametric_forecasters[best_model_name]
                model_kwargs = self._get_parametric_forecasters_kwargs(best_model_name, f"{target_dataset}")
            else:
                model_class = self.nonparametric_forecasters[best_model_name]
                model_kwargs = self._get_nonparametric_forecasters_kwargs(best_model_name, f"{target_dataset}")
            
            # Create and fit model on full dataset
            full_model = model_class(**model_kwargs)
            full_model.fit()
            
            # Generate future forecasts
            future_forecast = full_model.forecast(periods)
            
            # Convert back to original scale
            preprocessor = self.preprocessors[target_dataset]
            if hasattr(future_forecast, 'yhat'):  # Prophet
                future_values = future_forecast.yhat.values
            else:
                future_values = future_forecast.values if hasattr(future_forecast, 'values') else future_forecast
            
            future_original = preprocessor.inverse_transform(future_values)
            
            # Create future forecast dataframe with proper index
            last_date = full_data.index[-1]
            freq = full_data.index.freq or pd.infer_freq(full_data.index)
            future_index = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
            
            future_df = pd.DataFrame({
                f'{best_model_name}_future': future_original
            }, index=future_index)
            
            return future_df
            
        except Exception as e:
            print(f"Error generating future forecasts: {str(e)}")
            return None

    def run_experiment(self):
        print("Load Datasets")
        self.load_datasets()
        print("Split Datasets")
        self.split_train_test()
        print("Load Forecasters")
        self.load_forecasters()
        print("Define Fit and Save Models")
        self.define_fit_and_save_models()
        print("Forecast from models")
        self.forecast_from_models()
        print("Export experiment results")
        self.export_experiment_results()


# if __name__ == "__main__":
#     experimenter = Experimenter()
#     experimenter.load_datasets()
#     experimenter.get_full_data()
#     experimenter.load_forecasters()
#     experimenter.retrain_best_model_and_forecast_future("ets_forecaster", "tsa_checkpoint_travel_count_mon_weekly")
