import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from src.common.forecaster import Forecaster
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import pickle
import os

class ProphetForecaster(Forecaster):
    def __init__(self, data, name="prophet_model"):
        """
        Initialize the ProphetForecaster with time series data.

        Parameters:
        data (pd.Series): The time series data to be used for forecasting.
        name (str): The name for the model, used in paths for saving/loading.
        """
        super().__init__()
        self.data = data
        self.model = Prophet()
        self.metrics = None
        self.name = name
        self.path = self.get_cache_path(name)
        self.fitted_values = None
        self.forecast_values = None

    def prepare_data(self):
        """
        Prepare the data in Prophet's required format (columns 'ds' and 'y').
        """
        df = self.data.reset_index()
        df.columns = ['ds', 'y']
        return df

    def perform_randomized_search(self, df, param_grid, n_iter=50):
        """
        Faster implementation of randomized search for Prophet hyperparameter tuning.
        """
        # Initialize tracking
        all_metrics = []
        best_rmse = float('inf')
        best_params = None

        # Use a smaller validation set instead of full cross-validation
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        print("Starting hyperparameter tuning...")

        for i in range(n_iter):
            # Sample parameters
            params = {k: np.random.choice(v) for k, v in param_grid.items()}

            # Convert numpy types to Python native types
            params = {
                'changepoint_prior_scale': float(params['changepoint_prior_scale']),
                'seasonality_prior_scale': float(params['seasonality_prior_scale']),
                'holidays_prior_scale': float(params['holidays_prior_scale']),
                'seasonality_mode': str(params['seasonality_mode']),
                'changepoint_range': float(params['changepoint_range']),
                'n_changepoints': int(params['n_changepoints'])
            }

            try:
                # Fit model on training data
                model = Prophet(**params)
                model.fit(train_df)

                # Make predictions on validation set
                future = model.make_future_dataframe(periods=len(val_df), freq='W-MON')
                forecast = model.predict(future)

                # Calculate RMSE on validation set
                val_predictions = forecast.tail(len(val_df))
                rmse = np.sqrt(np.mean((val_predictions['yhat'].values - val_df['y'].values) ** 2))

                all_metrics.append({
                    'params': params,
                    'rmse': rmse
                })

                # Update best parameters if better RMSE found
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params

                print(f"Iteration {i + 1}/{n_iter}")
                print(f"Parameters: {params}")
                print(f"RMSE: {rmse}")
                print(f"Best RMSE so far: {best_rmse}\n")

            except Exception as e:
                print(f"Error with parameters {params}: {str(e)}")
                continue

        # Store results
        self.random_search_results = {
            'best_params': best_params,
            'best_score': best_rmse,
            'cv_results': all_metrics
        }

        print("\nBest parameters found:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        print(f"Best RMSE: {best_rmse}")

        # Create and return the best model
        best_model = Prophet(**best_params)
        best_model.fit(df)  # Fit on full dataset

        return best_model

    def fit(self):
        """
        Train the Prophet model on the provided data.
        """
        df = self.prepare_data()
        param_grid = {
            'changepoint_prior_scale': [0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.1, 1.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8],
            'n_changepoints': [20, 30]
        }

        # Perform random search
        self.model = self.perform_randomized_search(
            df=df,
            param_grid=param_grid,
            n_iter=50  # adjust based on your computational resources
        )
        self.save_search_results('Prophet')
        print("Model fitted successfully.")
        self.fitted_values = self.model.predict(df)

    def save_model(self, path=None):
        """
        Save the fitted Prophet model to a file using pickle.

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
        Load a previously saved Prophet model from a file using pickle.
        """
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {self.path}")
            df = self.prepare_data()
            self.fitted_values = self.model.predict(df)
        except FileNotFoundError:
            print(f"Model file not found at {self.path}. Fitting new model...")
            self.fit()


    def output(self):
        """
        Output the model's coefficients or other relevant information.
        """
        if self.model is not None:
            print(f"Model Coefficients: {self.model.params}")
        else:
            print("Model is not fitted yet.")

    def log_metrics(self):
        # Calculate and log training metrics if applicable
        if self.fitted_values is not None:
            y_true = self.prepare_data()['y']
            y_pred = self.fitted_values['yhat']
            self.metrics = {"MAPE": mean_absolute_percentage_error(y_true, y_pred)}
            print("Training Metrics:")
            for metric, value in self.metrics.items():
                print(f"{metric}: {value:.4f}")

    def forecast(self, steps):
        """
        Forecast future values using the trained Prophet model.

        Parameters:
        steps (int): The number of future steps to forecast.

        Returns:
        pd.DataFrame: Forecasted values including uncertainty intervals.
        """
        future = self.model.make_future_dataframe(periods=steps, freq="W-MON")
        forecast = self.model.predict(future)
        return forecast.tail(steps)
    
    def forecast_with_intervals(self, steps, alpha=0.05):
        """
        Forecast with confidence intervals.
        
        Parameters:
        steps (int): Number of steps to forecast
        alpha (float): Significance level (0.05 for 95% confidence)
        
        Returns:
        dict: {'forecast': Series, 'lower': Series, 'upper': Series}
        """
        future = self.model.make_future_dataframe(periods=steps, freq="W-MON")
        forecast = self.model.predict(future)
        forecast_tail = forecast.tail(steps)
        
        return {
            'forecast': forecast_tail['yhat'],
            'lower': forecast_tail['yhat_lower'],
            'upper': forecast_tail['yhat_upper']
        }

    def plot_fit_vs_actual(self, steps):
        """
        Plot the actual data, fitted values, and forecasted values.

        Parameters:
        steps (int): The number of future steps to forecast.
        """
        df = self.prepare_data()

        # Forecast future values (out-of-sample predictions)
        forecasted_values = self.forecast(steps)

        # Plotting
        plt.figure(figsize=(10, 6))

        # Plot actual data (solid black line)
        plt.plot(self.data.iloc[-50:], color='black', label='Actual')

        # Plot fitted values (solid blue line)
        fitted_values = self.model.predict(df)
        plt.plot(fitted_values['ds'], fitted_values['yhat'], color='blue', label='Fitted')

        # Plot forecasted values (dotted red line)
        plt.plot(forecasted_values['ds'], forecasted_values['yhat'], 'r--', label='Forecasted')

        mape = round(mean_absolute_percentage_error(df['y'], fitted_values['yhat']), 2)
        # Add titles and labels
        plt.title(f'{self.name}: Actual-Fit-Forecasts; Training MAPE: {mape}')
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{os.getcwd()}/plots/{self.name}_fit_forecast_plot.png")

# Example usage:
# prophet_forecaster = ProphetForecaster(data=data_series)
# prophet_forecaster.fit()
# prophet_forecaster.save_model('prophet_model.pkl')
# prophet_forecaster.plot_fit_vs_actual(steps=10)

# To load the model later:
# prophet_forecaster.load_model('prophet_model.pkl')
# forecasted_values = prophet_forecaster.forecast(steps=10)
# prophet_forecaster.plot_fit_vs_actual(steps=10)
