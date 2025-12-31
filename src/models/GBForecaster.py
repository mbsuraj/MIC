import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

class GBForecaster(Forecaster):
    def __init__(self, data, lags=10, name="gradient_boosting_model"):
        """
        Initialize the GradientBoostingForecaster with time series data and lags.

        Parameters:
        data (pd.Series): The time series data to be used for forecasting.
        lags (int): The number of lagged observations to use as features.
        name (str): The name for the model, used in paths for saving/loading.
        """
        super().__init__()
        self.random_search_results = None
        self.data = data
        self.lags = lags
        self.model = LGBMRegressor()
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.feature_data = None
        self.fitted_values = None
        self.forecast_values = None

    def create_features(self):
        """
        Generate lagged features and time-based features for the time series.
        """
        df = pd.DataFrame(self.data.values, columns=["value"], index=self.data.index)
        for i in range(1, self.lags + 1):
            df[f"lag_{i}"] = df["value"].shift(i)

        # Add time-based features
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Drop rows with NaN values (due to lagging)
        self.feature_data = df.dropna()
        print("Features created successfully.")

    def perform_randomized_search(self, X, y, param_grid, n_iter=50, eval_set=None,
                                  eval_metric='rmse', categorical_feature='auto', cv=5):
        """
        Perform randomized search for hyperparameter tuning.

        Parameters
        ----------
        X : array-like
            Training data features
        y : array-like
            Target values
        param_grid : dict
            Dictionary with parameters names (string) as keys and distributions
            or lists of parameters to try
        n_iter : int, default=100
            Number of parameter settings that are sampled
        eval_set : list of (X, y) tuples, optional
            Validation set for evaluation
        eval_metric : str or callable, default='rmse'
            Evaluation metric
        categorical_feature : list of str or int, or 'auto', default='auto'
            Categorical features
        cv : int, default=5
            Number of cross-validation folds

        Returns
        -------
        LGBMRegressor
            Best model found by random search
        """

        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            # random_state=42,
            n_jobs=-1,
            verbose=1,
            scoring='neg_mean_squared_error'
        )

        # Perform random search with LightGBM specific parameters
        random_search.fit(
            X,
            y,
            eval_set=eval_set,
            eval_metric=eval_metric,
            categorical_feature=categorical_feature
        )

        # Store results for later retrieval
        self.random_search_results = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'cv_results': random_search.cv_results_
        }

        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best cross-validation score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def get_random_search_results(self):
        """
        Get the results from the last random search.

        Returns
        --------
        dict
            Dictionary containing best parameters, best score, and CV results
        """
        if hasattr(self, 'random_search_results'):
            return self.random_search_results
        else:
            raise ValueError("No random search has been performed yet. "
                             "Run perform_randomized_search first.")

    def fit(self):
        """
        Fit LGBMRegressor with grid search.

        Parameters:
        -----------
        eval_set : list of (X, y) tuples, optional
            Validation set for evaluation
        eval_metric : str or callable, default='rmse'
            Metric used for evaluation
        categorical_feature : list of str or int, or 'auto', default='auto'
            Categorical features
        param_grid : dict, optional
            Custom parameter grid. If None, uses default LightGBM parameters
        """
        if self.feature_data is None:
            self.create_features()

        X = self.feature_data.drop("value", axis=1)
        y = self.feature_data["value"]

        # Default parameter grid for LightGBM
        param_grid = {
            'num_leaves': [15, 31, 50],  # reduced leaf count for smaller dataset
            'max_depth': [5, 8],  # more controlled depth
            'learning_rate': [0.05, 0.1],  # slightly higher learning rates
            'n_estimators': [50, 100, 150],  # reduced number of trees
            'min_child_samples': [10, 20],  # adjusted for dataset size
            'subsample': [0.8, 0.9],  # good range for smaller datasets
            'colsample_bytree': [0.8, 0.9],  # feature subsampling
            'reg_alpha': [0.1, 0.5],  # increased regularization
            'reg_lambda': [0.1, 0.5]  # increased regularization
        }

        # Perform grid search
        self.model = self.perform_randomized_search(
            X=X,
            y=y,
            param_grid=param_grid
        )

        # Final fit with best parameters
        self.fitted_values = self.model.predict(X)
        self.save_search_results('GradientBoosting')
        print("Model fitted successfully with optimal parameters.")

    def save_model(self, path=None):
        """
        Save the fitted Gradient Boosting model to a file using pickle.

        Parameters:
        path (str): The file path where the model should be saved.
        """
        if path is not None:
            self.path = path
        with open(self.path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {self.path}")

    def load_model(self):
        """
        Load a previously saved Gradient Boosting model from a file using pickle.
        """
        try:
            if self.feature_data is None:
                self.create_features()

            X = self.feature_data.drop("value", axis=1)

            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.path}")
            self.fitted_values = self.model.predict(X)
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.fit()

    def output(self):
        """
        Output the model's parameters and performance metrics.

        Prints the model's feature importances and, if the model has been fitted,
        displays key metrics such as training MAPE.
        """
        if self.feature_data is None:
            self.create_features()

        # Check if the model is fitted
        if hasattr(self.model, "feature_importances_"):
            print(
                f"Feature Importances:\n{pd.Series(self.model.feature_importances_, index=self.feature_data.columns.drop('value'))}")
        else:
            print("Model is not fitted yet.")

    def log_metrics(self):
        # Calculate and log training metrics if applicable
        if self.feature_data is not None:
            X = self.feature_data.drop("value", axis=1)
            y_true = self.feature_data["value"]
            if hasattr(self.model, "predict"):
                y_pred = self.model.predict(X)
                self.metrics = {"MAPE": mean_absolute_percentage_error(y_true, y_pred)}
                print("Training Metrics:")
                for metric, value in self.metrics.items():
                    print(f"{metric}: {value:.4f}")
    def forecast(self, steps):
        """
        Forecast future values using the trained Gradient Boosting model.

        Parameters:
        steps (int): The number of future steps to forecast.

        Returns:
        pd.Series: Forecasted values with a DateTimeIndex continuing from the end of the training data.
        """
        if self.feature_data is None:
            self.create_features()

        forecast_index = pd.date_range(start=self.data.index[-1], periods=steps + 1, freq="W-MON")[1:]
        forecasted_values = []

        last_known_data = self.feature_data.iloc[-1].drop("value")

        for _ in range(steps):
            forecast = self.model.predict([last_known_data])[0]
            forecasted_values.append(forecast)

            # Shift features for the next prediction
            new_row = pd.Series({
                f"lag_{i}": last_known_data[f"lag_{i - 1}"] if i > 1 else forecast
                for i in range(1, self.lags + 1)
            })
            new_row["month"] = forecast_index[0].month
            new_row["day_of_week"] = forecast_index[0].dayofweek
            new_row["day_of_month"] = forecast_index[0].day
            new_row["is_weekend"] = int(forecast_index[0].dayofweek >= 5)
            last_known_data = new_row

        return pd.Series(forecasted_values, index=forecast_index)

    def plot_fit_vs_actual(self, steps):
        """
        Plot the actual data, fitted values, and forecasted values.

        Parameters:
        steps (int): The number of future steps to forecast.
        """
        if self.feature_data is None:
            self.create_features()

        # Predict the fitted values (in-sample predictions)
        X = self.feature_data.drop("value", axis=1)
        y_true = self.feature_data["value"]
        fitted_values = self.model.predict(X)

        # Forecast future values (out-of-sample predictions)
        forecasted_values = self.forecast(steps)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot actual data (solid black line)
        plt.plot(self.data.iloc[-50:], color='black', label='Actual')

        # Plot fitted values (dotted yellow line)
        plt.plot(self.feature_data.index[-50:], fitted_values[-50:], 'y--', label='Fitted')

        # Plot forecasted values (dotted red line)
        plt.plot(forecasted_values, 'r--', label='Forecasted')

        mape = round(mean_absolute_percentage_error(y_true, fitted_values), 2)
        # Add titles and labels
        plt.title(f'{self.name}: Actual-Fit-Forecasts; Training MAPE: {mape}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{os.getcwd()}/plots/{self.name}_fit_forecast_plot.png")

# Example usage:
# gb_forecaster = GradientBoostingForecaster(data=data_series, lags=12)
# gb_forecaster.fit()
# gb_forecaster.save_model('gb_model.pkl')
# gb_forecaster.plot_fit_vs_actual(steps=10)

# To load the model later:
# gb_forecaster.load_model('gb_model.pkl')
# forecasted_values = gb_forecaster.forecast(steps=10)
# gb_forecaster.plot_fit_vs_actual(steps=10)
