import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import logging
import os
import json
from datetime import datetime, timedelta

from database import (
    db_config, HistoricalData, TechnicalIndicators, 
    ModelVersions, Predictions, DatabaseQueries, init_database
)

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

    
    def prepare_target(self, data, chunk_size=5):
        #data['Target'] = data['Adj Close'].shift(-1)
        for i in range(1, chunk_size + 1):
            data[f'Target_{i}'] = data['Adj Close'].shift(-i)
        return data

    def clean_data(self, data, columns_to_drop=['Volume', 'Close', 'Date']):
        data = data.copy()
        if data.index.name == 'Date' or 'Date' in data.columns:
            data.reset_index(inplace=True, drop=True)
        actual_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
        if actual_columns_to_drop:
            data.drop(actual_columns_to_drop, axis=1, inplace=True)
        data.dropna(subset=['Adj Close'], inplace = True)
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
    
    def prepare_lstm_data(self, data_set_scaled, backcandles=30, chunk_size=5, feature_columns=None):

        """
        Prepare data for LSTM model by creating sequences of historical data.

        Args:
        data_set_scaled (np.array): Scaled input data
        backcandles (int): Number of historical time steps to use for each sample
        chunk size: NUmber of target values to predict
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
        target_start_idx = data_set_scaled.shape[1] - chunk_size
        y = data_set_scaled[backcandles:, target_start_idx:]

        return np.array(X), y
    
    def create_and_train_lstm(self, X_train, y_train, chunk_size=5):
        """
        Creates and trains lstm model

        Args:
        X_train(np.ndarray): Training input data with shape (samples, backcandles, features)
        y_train(np.ndarray): training target data

        Returns:
        keras.Model: trained LSTM keras model

        """

        lstm_input = layers.Input(shape=(self.backcandles, len(self.feature_columns)), name='lstm_input')
        inputs = layers.LSTM(self.lstm_units, name='first_layer')(lstm_input)
        inputs = layers.Dense(128)(inputs)
        inputs = layers.Dense(chunk_size, name='dense_layer')(inputs)
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
        chunk_size = 5
        data = self.prepare_target(self.data.copy(), chunk_size)
        data = self.clean_data(data)
        print(f"Data shape after cleaning: {data.shape}")
        print(f"NaN values in data: {data.isna().sum().sum()}")
        if data.isna().sum().sum() > 0:
            print("Columns with NaN values:", data.columns[data.isna().any()].tolist())
            data = data.dropna()  # Drop any remaining NaN rows
            print(f"Data shape after dropping NaN: {data.shape}")
        feature_names = [col for col in data.columns if not col.startswith('Target')]
        target_names = [col for col in data.columns if col.startswith('Target')]

        feature_data = data[feature_names]
        target_data = data[target_names]

        feature_data_scaled, scaler = self.scale_data(feature_data)
        self.scaler = scaler

        #scale targets separately
        target_scaler = MinMaxScaler(feature_range=(0,1))
        target_data_scaled = target_scaler.fit_transform(target_data)
        self.target_scaler = target_scaler

        data_set_scaled = np.hstack([feature_data_scaled, target_data_scaled])

        self.feature_columns = list(range(len(feature_names)))
        
        X, y = self.prepare_lstm_data(
            data_set_scaled, 
            self.backcandles,
            chunk_size, 
            self.feature_columns
        )
        
        splitlimit = int(len(X)*0.8)
        X_train, X_test = X[:splitlimit], X[splitlimit:]
        y_train, y_test = y[:splitlimit], y[splitlimit:]
        
        model, history = self.create_and_train_lstm(X_train, y_train, chunk_size)
        return model, history, X_test, y_test

    def predict(self, data, chunk_size=5):
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Use the same data preparation pipeline as training
        data = self.prepare_target(data.copy(), chunk_size)
        data = self.clean_data(data)
        data_set_scaled, _ = self.scale_data(data, save_scaler=False)
        
        X, y = self.prepare_lstm_data(
            data_set_scaled, 
            self.backcandles, 
            chunk_size,
            self.feature_columns
        )
        
        predictions_scaled = self.model.predict(X)
        
        predictions = []
        for i in range(len(predictions_scaled)):
            sample_preds = []
            for j in range(chunk_size):
                # Inverse transform predictions
                dummy = np.zeros((1, self.scaler.n_features_in_))
                target_col_idx = len(self.feature_columns) + j
                dummy[0, target_col_idx] = predictions_scaled[i,j]
                predict = self.scaler.inverse_transform(dummy)[0, target_col_idx]
                sample_preds.append(predict)
            predictions.append(sample_preds)
        
        
        self.last_predictions = predictions
        self.last_actual = y
        self.last_X = X
        self.last_y = y

        return predictions
    
    def predict_next(self, recent_data):
        """Predict just next timestep"""
        chunk_predictions = self.predict_next_chunk(recent_data, chunk_size=1)
        return chunk_predictions[0]
    
    def predict_next_chunk(self, recent_data, chunk_size=5):
        """Predict next chunk of prices"""
        if len(recent_data) < self.backcandles + 1:
            raise ValueError(f"Need {self.backcandles + 1} candles, got {len(recent_data)}")
        
        data = recent_data.copy()
        data = self.clean_data(data)
        if len(data) < self.backcandles:
            raise ValueError(f"Not enough data after cleaning: {len(data)}")
        
        feature_names = [col for col in data.columns if not col.startswith('Target')]
        feature_data = data[feature_names]
        data_scaled = self.scaler.transform(feature_data)
        

        X = data_scaled[-self.backcandles:, self.feature_columns]
        X = X.reshape(1, self.backcandles, len(self.feature_columns))

        predictions_scaled = self.model.predict(X)

        predictions = self.target_scaler.inverse_transform(predictions_scaled.reshape(1,-1))[0]

        return predictions.tolist()
    
    def predict_next_chunk_metrics(self, recent_data, chunk_size=5):
        """Predict useful trading metrics for next chunk"""
        try:
            predictions = self.predict_next_chunk(recent_data, chunk_size)
            current_price = recent_data['Adj Close'].iloc[-1]

            if not predictions:
                return None
            
            if hasattr(self, 'model_id') and self.model_id:
                with db_config.get_db_session() as session:
                    predictions_data = []
                    for i, pred_price in enumerate(predictions):
                        predictions_data.append({
                            'target_date': datetime.now() + timedelta(minutes=i+1),
                            'predicted_value': pred_price,
                            'confidence_score': None
                        })

                    DatabaseQueries.store_predictions(
                        session, self.model_id, self.ticker, predictions_data
                    )
            
            return {
                'raw_predictions': predictions,
                'current_price': current_price,
                'mean_price': np.mean(predictions),
                'max_price': np.max(predictions),
                'min_price': np.min(predictions),
                'final_price': predictions[-1],
                'direction': 'UP' if predictions[-1] > current_price else 'DOWN',
                'price_change_pct': ((predictions[-1] - current_price) / current_price) * 100,
                'volatility': np.std(predictions),
                'trend_strength': (predictions[-1] - predictions[0]) / predictions[0] * 100 if predictions[0] != 0 else 0,
                'momentum': 'INCREASING' if len(predictions) > 2 and predictions[-1] > predictions[len(predictions)//2] else 'DECREASING'
            }
        except Exception as e:
            print(f'Error in predict_chunk_metrics: {str(e)}')
            return None
    


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
        joblib.dump(self.target_scaler, f"{version_path}target_scaler.pkl")

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

        with db_config.get_db_session() as session:
            model_version = ModelVersions(
                version = self.version,
                parameters=json.dumps({
                    'backcandles': self.backcandles,
                    'lstm_units': self.lstm_units,
                    'feature_columns': self.feature_columns,
                    'ticker': self.ticker
                }),
                metrics = json.dumps(self.training_metadata),
                is_active = True
            )

            session.query(ModelVersions).filter(
                ModelVersions.parameters.like(f'%"ticker": "{self.ticker}"%')
            ).update({'is_active': False})

            session.add(model_version)
            session.flush()

            self.model_id = model_version.model_id
        
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
            target_scaler_path = os.path.join(version_path, "target_scaler.pkl")
            metadata_path = os.path.join(version_path, "metadata.json")

            self.model = models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.target_scaler = joblib.load(target_scaler_path)

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