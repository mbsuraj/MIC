import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, data):
        self.original_data = data
        self.scaler = StandardScaler()
        self.scaled_data = self._scale_data()
    
    def _scale_data(self):
        scaled = self.scaler.fit_transform(self.original_data.values.reshape(-1, 1)).flatten()
        return pd.Series(scaled, index=self.original_data.index)
    
    def inverse_transform(self, values):
        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()