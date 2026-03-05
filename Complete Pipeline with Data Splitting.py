import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Conv1D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class NuclearAnomalyDetectionPipeline:
    """
    Complete pipeline for nuclear power plant anomaly detection
    """
    
    def __init__(self, window_size=50, threshold_std=3):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None
        
    def load_and_preprocess(self, filepath):
        """Load CSV and normalize data"""
        df = pd.read_csv(filepath)
        
        # Extract features and labels
        X = df.iloc[:, 1:-1].values  # All sensor columns
        y = df.iloc[:, -1].values     # Label column
        
        # Normalize
        X_normalized = self.scaler.fit_transform(X)
        
        print(f"Raw data shape: {X_normalized.shape}")
        return X_normalized, y
    
    def create_windows(self, X, y):
        """Create sliding windows"""
        X_windows = []
        y_windows = []
        
        for i in range(len(X) - self.window_size):
            X_windows.append(X[i:i + self.window_size])
            y_windows.append(y[i + self.window_size])
        
        return np.array(X_windows), np.array(y_windows)
    
    def split_train_test(self, X_windows, y_windows, test_size=0.2, random_state=42):
        """
        Split into train and test
        Training uses only normal samples
        """
        # Get all indices
        all_indices = np.arange(len(X_windows))
        normal_indices = np.where(y_windows == 0)[0]
        anomaly_indices = np.where(y_windows == 1)[0]
        
        # Split normal data for training
        normal_train_idx, normal_test_idx = train_test_split(
            normal_indices, test_size=test_size, random_state=random_state
        )
        
        # Combine normal test with all anomalies for comprehensive testing
        test_idx = np.concatenate([normal_test_idx, anomaly_indices])
        
        X_train = X_windows[normal_train_idx]
        y_train = y_windows[normal_train_idx]
        X_test = X_windows[test_idx]
        y_test = y_windows[test_idx]
        
        print(f"Training set: {X_train.shape} (normal only)")
        print(f"Test set: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, n_features):
        """Build autoencoder"""
        model = Sequential([
            Conv1D(filters=32, kernel_size=3, padding='same', 
                   activation='relu', input_shape=(self.window_size, n_features)),
            LSTM(64, activation='relu', return_sequences=False),
            RepeatVector(self.window_size),
            LSTM(64, activation='relu', return_sequences=True),
            Dense(n_features, activation='linear')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model = model
        return model
    
    def train(self, X_train, epochs=50, batch_size=32):
        """Train the model"""
        history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        return history
    
    def calculate_errors(self, X_data):
        """Calculate reconstruction errors"""
        X_pred = self.model.predict(X_data, verbose=0)
        mse = np.mean(np.square(X_data - X_pred), axis=(1, 2))
        return mse
    
    def set_threshold(self, train_errors):
        """Set anomaly threshold"""
        mean_error = np.mean(train_errors)
        std_error = np.std(train_errors)
        self.threshold = mean_error + self.threshold_std * std_error
        
        print(f"Mean error: {mean_error:.4f}")
        print(f"Std error: {std_error:.4f}")
        print(f"Threshold: {self.threshold:.4f}")
        
        return self.threshold
    
    def predict(self, test_errors):
        """Predict anomalies"""
        return (test_errors > self.threshold).astype(int)
    
    def evaluate(self, y_true, y_pred):
        """Evaluate performance"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }


def run_pipeline():
    """Execute complete pipeline"""
    pipeline = NuclearAnomalyDetectionPipeline(window_size=50, threshold_std=3)
    
    # Step 1-2: Load and preprocess
    X, y = pipeline.load_and_preprocess('SWaT_modified_8_features_dataset.csv')
    
    # Step 3-4: Create windows
    X_windows, y_windows = pipeline.create_windows(X, y)
    print(f"Windowed data shape: {X_windows.shape}")
    
    # Step 5-6: Split data
    X_train, y_train, X_test, y_test = pipeline.split_train_test(X_windows, y_windows)
    
    # Step 7: Build model
    pipeline.build_model(n_features=X.shape[1])
    
    # Step 8: Train
    history = pipeline.train(X_train, epochs=50)
    
    # Step 9: Calculate errors
    train_errors = pipeline.calculate_errors(X_train)
    test_errors = pipeline.calculate_errors(X_test)
    
    # Step 10: Set threshold
    pipeline.set_threshold(train_errors)
    
    # Step 11: Predict
    y_pred = pipeline.predict(test_errors)
    
    # Step 12: Evaluate
    metrics = pipeline.evaluate(y_test, y_pred)
    
    return pipeline, history, train_errors, test_errors, y_test, y_pred


if __name__ == "__main__":
    pipeline, history, train_errors, test_errors, y_test, y_pred = run_pipeline()