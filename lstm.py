import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import logging
import os
import json
from datetime import datetime

class StockPredictor:
    def __init__(self):
        #Default parameters
        self.version = "1.0.0"
        self.training_metadata = {}
        self.backcandles = 7
        self.target_column = -1
        self.feature_columns = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        self.lstm_units = 100
        self.batch_size = 12
        self.epochs = 300
        self.validation_split = 0.1
        self.patience = 15
        self.model = None
        self.scaler = None

    
    def get_ticker_data(self, TICKER, START_DATE='2014-08-01', END_DATE='2024-08-01'):
        # Your existing get_ticker_data function
        try:
            data = yf.download(TICKER, start=START_DATE, end=END_DATE)
            if data.empty:
                raise ValueError(f"No data found for {TICKER}")
            return data
        except Exception as e:
            print(f"Error: {e}")
            raise

    def add_indicators(self, data, indicator_set='default'):
        """
        Add technical indicators to the dataset using native pandas calculations
        where possible to avoid dependency issues.
        """
        if indicator_set == 'default':
            # RSI calculation
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=15).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=15).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # EMA calculations
            data['EMAF'] = data['Close'].ewm(span=20, adjust=False).mean()

            # Historical Volatility
            data['hist_volatility'] = data['Adj Close'].pct_change().rolling(window=20).std() * np.sqrt(252)

            # Bollinger Bands
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['BBL_20_2.0'] = rolling_mean - (2 * rolling_std)
            data['BBM_20_2.0'] = rolling_mean
            data['BBU_20_2.0'] = rolling_mean + (2 * rolling_std)
            data['BB_width'] = (data['BBU_20_2.0'] - data['BBL_20_2.0']) / data['BBM_20_2.0']

            # MACD
            exp1 = data['Close'].ewm(span=12, adjust=False).mean()
            exp2 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp1 - exp2
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # ATR
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            data['ATR'] = true_range.rolling(14).mean()

            # OBV
            data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

        elif indicator_set == 'alternative':
            data['EMA'] = data['Close'].ewm(span=50, adjust=False).mean()
            # VWAP calculation requires intraday data, removed for daily timeframe

        else:
            raise ValueError("Invalid indicator set. Use 'default' or 'alternative'.")

        return data
    
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
        for j in feature_columns:
            X.append([])
            for i in range(backcandles, data_set_scaled.shape[0]):
                X[j].append(data_set_scaled[i-backcandles:i, j])

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
    
    def train(self, TICKER, START_DATE='2014-08-01', END_DATE='2024-08-01'):
        # Main training pipeline
        data = self.get_ticker_data(TICKER, START_DATE, END_DATE)
        data = self.add_indicators(data)
        data = self.prepare_target(data)
        data = self.clean_data(data)
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

    def predict(self, TICKER, START_DATE, END_DATE):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Use the same data preparation pipeline as training
        data = self.get_ticker_data(TICKER, START_DATE, END_DATE)
        data = self.add_indicators(data)
        data = self.prepare_target(data)
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
        version_path = f"{path}v{self.version}_{timestamp}/"

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
                versions = sorted(os.listdir(path))
                if not versions:
                    raise ValueError("No models found")
                version_path = os.path.join(path, versions[-1])
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

            return self.training_metadata
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise