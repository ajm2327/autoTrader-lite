import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import logging
import os
import json
from datetime import datetime

class StockPredictor:
    def __init__(self, data, ticker=None):
        #Default parameters
        self.version = "1.0.0"
        self.ticker = ticker
        self.data = data
        self.training_metadata = {}
        self.backcandles = 20
        self.target_column = -1
        # Features: Open, High, Low, vwap, Adj Close, SMA_20, SMA_50, SMA_200, RSI, MACD, Signal_Line, Middle_Band, Upper_Band, Lower_Band, EMA
        self.feature_columns = None#[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15]
        self.lstm_units = 100
        self.batch_size = 12
        self.epochs = 50
        self.validation_split = 0.1
        self.patience = 5
        self.model = None
        self.scaler = None

    
    def prepare_target(self, data):
        data['Target'] = data['Adj Close'].shift(-1)
        return data

    def clean_data(self, data, columns_to_drop=['Volume', 'Close', 'Date']):
        data.dropna(inplace=True)
        data.reset_index(inplace=True)
        data.drop(columns_to_drop, axis=1, inplace=True)
        return data
    
    def scale_data(self, data, feature_range=(0,1), save_scaler=True, scaler_path='scaler.pkl'):
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_columns]
        scaler = MinMaxScaler(feature_range=feature_range)
        data_scaled = scaler.fit_transform(data_numeric)
        if save_scaler:
            joblib.dump(scaler, scaler_path)
        self.scaler = scaler
        return data_scaled, scaler
    
    def prepare_lstm_data(self, data_set_scaled, backcandles=30, target_column=-1, feature_columns=None):

        """
        Prepare data for LSTM model by creating sequences of historical data.

        Args:
        data_set_scaled (np.array): Scaled input data
        backcandles (int): Number of historical time steps to use for each sample
        target_column (int): Index of the target column (-1 for last column)
        feature_columns (list): Number of columns to use as features (N)

        Returns:
        tuple: X (input sequences), y (target values)
        """

        X = []
        for idx, j in enumerate(feature_columns):
            X.append([])
            for i in range(backcandles, data_set_scaled.shape[0]):
                X[idx].append(data_set_scaled[i-backcandles:i, j])

        #Move axis from 0 to position 2
        X = np.moveaxis(X, [0], [2])
        #Extract target values
        y = data_set_scaled[backcandles:, target_column]

        return np.array(X), np.array(y).reshape(-1,1)
    
    def create_and_train_lstm(self, X_train, y_train):
        """
        Creates and trains lstm model

        Args:
        X_train(np.ndarray): Training input data with shape (samples, backcandles, features)
        y_train(np.ndarray): training target data
        backcandles(int): Number of time steps to look back (default = 30)
        features(int): Number of input features (default = 9)
        lstm_units(int): Number of units in LSTM layer (default = 150)
        batch_size(int): Batch size for training (default = 15)
        epochs(int): Number of epochs for training (default = 30)
        validation_split(float): fraction of training data to use for validation (default = 0.1)

        Returns:
        keras.Model: trained LSTM keras model

        """

        lstm_input = layers.Input(shape=(self.backcandles, len(self.feature_columns)), name='lstm_input')
        inputs = layers.LSTM(self.lstm_units, name='first_layer')(lstm_input)
        inputs = layers.Dense(128)(inputs)
        inputs = layers.Dense(1, name='dense_layer')(inputs)
        output = layers.Activation('linear', name='output')(inputs)
        model = models.Model(inputs=lstm_input, outputs=output)

        adam = optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=adam, loss='mse')

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            validation_split=self.validation_split,
            callbacks=[early_stopping, model_checkpoint]
        )

        self.model = model
        return model, history
    
    def train(self):
        # Main training pipeline
        data = self.prepare_target(self.data.copy())
        data = self.clean_data(data)
        feature_names = [col for col in data.columns if col != 'Target']
        self.feature_columns = list(range(len(feature_names)))

        data_set_scaled, scaler = self.scale_data(data)
        
        X, y = self.prepare_lstm_data(
            data_set_scaled, 
            self.backcandles, 
            self.target_column, 
            self.feature_columns
        )
        
        splitlimit = int(len(X)*0.8)
        X_train, X_test = X[:splitlimit], X[splitlimit:]
        y_train, y_test = y[:splitlimit], y[splitlimit:]
        
        model, history = self.create_and_train_lstm(X_train, y_train)
        return model, history, X_test, y_test

    def predict(self, data):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Use the same data preparation pipeline as training
        data = self.prepare_target(data.copy())
        data = self.clean_data(data)
        data_set_scaled, _ = self.scale_data(data, save_scaler=False)
        
        X, y = self.prepare_lstm_data(
            data_set_scaled, 
            self.backcandles, 
            self.target_column, 
            self.feature_columns
        )
        
        predictions_scaled = self.model.predict(X)
        
        # Inverse transform predictions
        dummy = np.zeros((len(predictions_scaled), self.scaler.n_features_in_))
        dummy[:, self.target_column] = predictions_scaled.flatten()
        predictions = self.scaler.inverse_transform(dummy)[:, self.target_column]
        
        
        self.last_predictions = predictions
        self.last_actual = y #store actual values
        self.last_X = X
        self.last_y = y

        return predictions

    def save_model(self, path='models_saved/'):
        if self.model is None:
            raise ValueError("No model to save")
        
        #create version specific filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        version_path = f"{path}{self.ticker}v{self.version}_{timestamp}/"

        #create directory if it doesn't exist
        os.makedirs(version_path, exist_ok = True)

        #Save model and associated files
        self.model.save(f"{version_path}lstm_model.keras")
        joblib.dump(self.scaler, f"{version_path}scaler.pkl")

        #save metadata
        self.training_metadata.update({
            'version': self.version,
            'timestamp': timestamp,
            'model_params': {
                'backcandles': self.backcandles,
                'lstm_units': self.lstm_units,
                'feature_columns': self.feature_columns
            }
        })

        with open(f"{version_path}metadata.json", 'w') as f:
            json.dump(self.training_metadata, f)

        return version_path

    def load_model(self, version=None, path='models_saved/'):
        try:
            if version is None:
                #Load latest version
                ticker_models = [d for d in os.listdir(path) if d.startswith(f'{self.ticker}v')]
                if not ticker_models:
                    raise ValueError("No models found for {self.ticker}")
                version_path = os.path.join(path, sorted(ticker_models)[-1])
            else:
                #load specific version
                matching_dirs = [d for d in os.listdir(path) if d.startswith(f'v{version}_')]
                if not matching_dirs:
                    raise ValueError(f"Version {version} not found")
                
                version_dir = sorted(matching_dirs)[-1]
                version_path = os.path.join(path, version_dir)

            
            model_path = os.path.join(version_path, 'lstm_model.keras')
            scaler_path = os.path.join(version_path, "scaler.pkl")
            metadata_path = os.path.join(version_path, "metadata.json")

            self.model = models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)

            #load metadata
            with open(metadata_path, 'r') as f:
                self.training_metadata = json.load(f)
            self.version = self.training_metadata['version']

            if 'model_params' in self.training_metadata:
                params = self.training_metadata['model_params']
                self.backcandles = params.get('backcandles', self.backcandles)
                self.lstm_units = params.get('lstm_units', self.lstm_units)
                self.feature_columns = params.get('feature_columns', self.feature_columns)

            return self.training_metadata
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise