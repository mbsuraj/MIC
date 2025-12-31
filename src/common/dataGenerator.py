import pandas as pd
import os
import json


class DataGenerator:
    def __init__(self, directory, config_path):
        """
        Initializes the DataGenerator class.

        Parameters:
        directory (str): Path to the directory containing CSV files.
        config_path (str): Path to the JSON configuration file for date parsing.
        """
        self.all_data_dict = dict()
        self._directory = directory
        self._config_path = config_path
        self._config = self._load_config()

    def _load_config(self):
        """
        Load the JSON configuration file for date parsing options.

        Returns:
        dict: A dictionary containing the configurations from the JSON file.
        """
        try:
            with open(self._config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            print(f"Configuration file not found at {self._config_path}. Using default settings.")
            return {}

    def _clean_data(self, df):
        """
        Clean and sort the DataFrame by date and reset the index.

        Parameters:
        df (pd.DataFrame): The DataFrame to clean.

        Returns:
        pd.DataFrame: The cleaned DataFrame.
        """
        df = df.reset_index(drop=True)
        return df

    def _clean_columns(self, df):
        """
        Clean column names by converting to lowercase and removing spaces.

        Parameters:
        df (pd.DataFrame): The DataFrame to clean.

        Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
        """
        df.columns = [c.lower().replace(" ", "") for c in df.columns]
        return df

    def _setup_datetime(self, df, dataset_name):
        """
        Set up the 'date' column as a datetime index using parameters from the configuration.

        Parameters:
        df (pd.DataFrame): The DataFrame with a 'date' column.
        dataset_name (str): The name of the dataset, used to find specific settings in the configuration.

        Returns:
        pd.DataFrame: The DataFrame with the 'date' column converted to datetime.
        """
        if dataset_name in self._config:
            date_config = self._config[dataset_name]
            format_str = date_config.get("format", "%Y-%m-%d")  # Default format
            freq = date_config.get("freq", None)  # Optional frequency setting
            df["date"] = pd.to_datetime(df["date"], format=format_str)

            # # Optional: If freq is provided in the config, use it to set the index frequency
            # if freq:
            #     df = df.asfreq(freq)

        else:
            # Default behavior if no config exists for the dataset
            df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
        df.set_index("date", inplace=True)
        df = df.sort_index()
        date_df = pd.DataFrame(index=pd.date_range(start=df.index[0], end=df.index[-1], freq=freq))
        df = pd.merge(date_df, df, left_index=True, right_index=True, how="left")
        df = df.ffill()
        return df

    def load_data(self):
        """
        Load all CSV files in the specified directory, clean them, and parse dates.

        Returns:
        dict: A dictionary containing the cleaned DataFrames with date indexes.
        """
        self._read_csv_files()
        for dataset_name, df in self.all_data_dict.items():
            print(f"Processing dataset: {dataset_name}")
            df = self._clean_columns(df)
            df = self._clean_data(df)
            df = self._setup_datetime(df, dataset_name)
            print(df.head())
            self.all_data_dict[dataset_name] = df
        return self.all_data_dict

    def _read_csv_files(self):
        """
        Read all CSV files in the directory into the data dictionary.
        """
        for filename in os.listdir(self._directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(self._directory, filename)
                df = pd.read_csv(filepath)
                dataset_name = filename.replace(".csv", "")
                self.all_data_dict[dataset_name] = df
