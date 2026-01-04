import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

class GBForecaster(Forecaster):
    def __init__(self, data, lags=10, name="gradient_boosting_model", data_freq='W-Mon'):
        """
        Initialize the GradientBoostingForecaster with time series data and lags.

        Parameters:
        data (pd.Series): The time series data to be used for forecasting.
        lags (int): Unused - kept for compatibility. Lags are now generated based on data_freq.
        name (str): The name for the model, used in paths for saving/loading.
        """
        super().__init__()
        self.random_search_results = None
        self.data = data
        self.data_freq = data_freq
        self.lag_periods = self._get_frequency_lags(data_freq)
        self.model = LGBMRegressor()
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.feature_data = None
        self.fitted_values = None
        self.forecast_values = None

    def _get_frequency_lags(self, data_freq):
        """Generate frequency-specific lag periods."""
        freq_map = {
            'D': [1, 2, 3, 4, 5, 6, 7, 30, 365],  # Daily: recent days + monthly + yearly
            'W': [1, 2, 3, 4, 52],  # Weekly: recent weeks + yearly
            'W-MON': [1, 2, 3, 4, 52],  # Weekly Monday: recent weeks + yearly
            'M': [1, 2, 3, 12],  # Monthly: recent months + yearly
            'Q': [1, 2, 4],  # Quarterly: recent quarters + yearly
            'Y': [1, 2, 3]  # Yearly: recent years
        }
        return freq_map.get(data_freq, [1, 2, 3, 4, 5])  # Default fallback

    def create_features(self):
        """
        Generate frequency-aware lagged features and time-based features.
        """
        df = pd.DataFrame(self.data.values, columns=["value"], index=self.data.index)
        
        # Create frequency-specific lag features
        for lag in self.lag_periods:
            df[f"lag_{lag}"] = df["value"].shift(lag)

        # Add time-based features
        df["month"] = df.index.month
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Drop rows with NaN values (due to lagging)
        self.feature_data = df.dropna()
        print(f"Features created with lags: {self.lag_periods}")

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

    def fit(self, params):
        """
        Fit the Gradient Boosting model with given parameters.
        """
        if self.feature_data is None:
            self.create_features()
        
        X = self.feature_data.drop("value", axis=1)
        y = self.feature_data["value"]
        
        self.model = LGBMRegressor(**params)
        self.model.fit(X, y)
        self.fitted_values = self.model.predict(X)

    def search_and_fit(self):
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
        self.save_search_results('gb_forecaster')
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
            self.search_and_fit()

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

        # Create future date index
        forecast_index = pd.date_range(
            start=self.data.index[-1],
            periods=steps + 1,
            freq=self.data_freq
        )[1:]

        forecasted_values = []
        
        # Maintain history for lag features (extend original data with forecasts)
        extended_data = self.data.copy()
        
        for step in range(steps):
            # Create features for current step
            current_features = self._create_forecast_features(extended_data, forecast_index[step])
            
            # Make prediction
            forecast_value = self.model.predict([current_features])[0]
            forecasted_values.append(forecast_value)
            
            # Extend data with new forecast for next iteration
            extended_data = pd.concat([
                extended_data, 
                pd.Series([forecast_value], index=[forecast_index[step]])
            ])

        return pd.Series(forecasted_values, index=forecast_index)

    def _create_forecast_features(self, data, date):
        """Create features for a single forecast step."""
        features = {}
        
        # Create lag features
        for lag in self.lag_periods:
            if len(data) >= lag:
                features[f"lag_{lag}"] = data.iloc[-lag]
            else:
                features[f"lag_{lag}"] = 0  # Fallback for insufficient history
        
        # Add time-based features
        features["month"] = date.month
        features["day_of_week"] = date.dayofweek
        features["day_of_month"] = date.day
        features["is_weekend"] = int(date.dayofweek >= 5)
        
        return pd.Series(features)



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
