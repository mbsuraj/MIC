from abc import ABC, abstractmethod
import json
from datetime import datetime
import os

# Get absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

class Forecaster(ABC):
    def __init__(self):
        self.name = None
        self.data = None
        self.random_search_results = None

    def get_cache_path(self, name):
        """Get absolute path for model cache"""
        cache_dir = os.path.join(PROJECT_ROOT, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{name}.pkl")

    def save_search_results(self, model_name):
        """
        Save random search results to a JSON file with timestamp

        Parameters
        ----------
        model_name : str
            Name of the model (e.g., 'Prophet', 'RandomForest', 'GradientBoosting', 'SVM', 'NeuralNetwork')
        """
        # Create results directory if it doesn't exist
        results_dir = os.path.join(PROJECT_ROOT, 'model_search_results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Prepare the results dictionary
        results_entry = {
            'date': timestamp,
            'model': model_name,
            'best_params': self.random_search_results['best_params'],
            'best_score': float(self.random_search_results['best_score']),  # Convert numpy types to native Python types
            'dataset_info': {
                'n_samples': len(self.data),
                'dataset_name_and_type': self.name,
            }
        }

        # File path for the JSON
        json_file = os.path.join(results_dir, 'hyperparameter_search_results.json')

        try:
            # Read existing results if file exists
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    existing_results = json.load(f)
            else:
                existing_results = []

            # Append new results
            existing_results.append(results_entry)

            # Write back to file
            with open(json_file, 'w') as f:
                json.dump(existing_results, f, indent=4)

            print(f"Search results saved successfully to {json_file}")
        except json.JSONDecodeError:
            # Append new results
            existing_results = []
            existing_results.append(results_entry)
            # Write back to file
            with open(json_file, 'w') as f:
                json.dump(existing_results, f, indent=4)
                print(f"Search results saved successfully to {json_file}")
        except Exception as e:
            print(f"Error saving results to JSON: {str(e)}")

    @abstractmethod
    def fit(self):
        """
        Fit the model on the provided data.
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self):
        """
        Log relevant metrics after training the model or during evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path):
        """
        Save the trained model to a file.

        Parameters:
        path (str): The file path where the model should be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path):
        """
        Load a previously saved model from a file.

        Parameters:
        path (str): The file path from which the model should be loaded.
        """
        raise NotImplementedError

    @abstractmethod
    def output(self):
        """
        Output the model's parameters, coefficients, or any other relevant information.
        """
        raise NotImplementedError

    @abstractmethod
    def forecast(self, steps):
        """
        Forecast future values using the trained model.

        Parameters:
        steps (int): The number of future steps to forecast.
        """
        raise NotImplementedError

    @abstractmethod
    def plot_fit_vs_actual(self, steps):
        """
        Plot the actual data, fitted values, and forecasted values.

        Parameters:
        steps (int): The number of future steps to forecast.
        """
        raise NotImplementedError
