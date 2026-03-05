import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Conv1D
from tensorflow.keras.optimizers import Adam
import seaborn as sns

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

def load_data(filepath):
    """
    Load the CSV dataset
    """
    df = pd.read_csv(filepath)
    return df


# ============================================================================
# STEP 2: DROP TIMESTAMP & NORMALIZE
# ============================================================================

def preprocess_data(df):
    """
    Step 2: Remove timestamp and normalize sensor values
    """
    # Drop timestamp column (first column)
    X = df.iloc[:, 1:-1].values  # All columns except timestamp and label
    y = df.iloc[:, -1].values     # Last column is label
    
    # Normalize to [0, 1] range
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y, scaler


# ============================================================================
# STEP 3: SLIDING WINDOW TRANSFORMATION
# ============================================================================

def create_sliding_windows(X, y, window_size=50):
    """
    Step 4: Create sliding windows
    
    Input shape: (N, features)
    Output shape: (num_windows, window_size, features)
    """
    X_windows = []
    y_windows = []
    
    for i in range(len(X) - window_size):
        # Take 50 consecutive rows
        window = X[i:i + window_size]
        X_windows.append(window)
        
        # Label comes from the LAST row in the window
        y_windows.append(y[i + window_size])
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    print(f"X_windows shape: {X_windows.shape}")
    print(f"y_windows shape: {y_windows.shape}")
    
    return X_windows, y_windows


# ============================================================================
# STEP 5: SEPARATE NORMAL AND ANOMALOUS DATA
# ============================================================================

def separate_normal_anomaly(X_windows, y_windows):
    """
    Step 6: Training data should only contain NORMAL behavior (label=0)
    """
    # Get indices where label == 0 (normal)
    normal_indices = np.where(y_windows == 0)[0]
    
    # Training data: only normal windows
    X_train = X_windows[normal_indices]
    y_train = y_windows[normal_indices]
    
    print(f"X_train (normal only) shape: {X_train.shape}")
    print(f"Removed {len(y_windows) - len(y_train)} anomalous windows from training")
    
    return X_train, y_train


# ============================================================================
# STEP 6: BUILD CONV1D-LSTM AUTOENCODER
# ============================================================================

def build_autoencoder(window_size=50, n_features=8):
    """
    Step 7: Build Conv1D-LSTM Autoencoder
    
    Architecture:
    - Conv1D: Detects short-term temporal patterns
    - LSTM Encoder: Compresses sequence to 64 numbers
    - RepeatVector: Copies 64 numbers 50 times
    - LSTM Decoder: Reconstructs temporal structure
    - Dense: Restores original sensor format
    """
    model = Sequential([
        # Encoder
        Conv1D(filters=32, kernel_size=3, padding='same', 
               activation='relu', input_shape=(window_size, n_features)),
        
        LSTM(64, activation='relu', return_sequences=False),
        
        # Bottleneck
        RepeatVector(window_size),
        
        # Decoder
        LSTM(64, activation='relu', return_sequences=True),
        
        # Output
        Dense(n_features, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    
    return model


# ============================================================================
# STEP 7: TRAIN THE MODEL
# ============================================================================

def train_model(model, X_train, epochs=50, batch_size=32):
    """
    Step 8: Train the autoencoder on normal data
    
    Model learns to reconstruct normal patterns with low error
    """
    history = model.fit(
        X_train, X_train,  # Input and output are the same (autoencoder)
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    return history


# ============================================================================
# STEP 8: CALCULATE RECONSTRUCTION ERRORS
# ============================================================================

def calculate_reconstruction_error(model, X_data):
    """
    Step 9: Calculate MSE between original and reconstructed
    """
    X_pred = model.predict(X_data, verbose=0)
    
    # MSE per sample
    mse_per_sample = np.mean(np.square(X_data - X_pred), axis=(1, 2))
    
    return mse_per_sample


# ============================================================================
# STEP 9: DETERMINE THRESHOLD
# ============================================================================

def determine_threshold(train_errors, threshold_std=3):
    """
    Step 10: Calculate threshold using mean + std*3
    
    This captures ~99.7% of normal data
    Anything beyond is considered anomalous
    """
    mean_error = np.mean(train_errors)
    std_error = np.std(train_errors)
    threshold = mean_error + threshold_std * std_error
    
    print(f"Mean training error: {mean_error:.4f}")
    print(f"Std training error: {std_error:.4f}")
    print(f"Threshold (mean + 3*std): {threshold:.4f}")
    
    return threshold, mean_error, std_error


# ============================================================================
# STEP 10: MAKE PREDICTIONS
# ============================================================================

def predict_anomalies(test_errors, threshold):
    """
    Step 11: Classify based on threshold
    
    If error > threshold → anomaly (1)
    Else → normal (0)
    """
    predictions = (test_errors > threshold).astype(int)
    return predictions


# ============================================================================
# STEP 11: EVALUATE MODEL
# ============================================================================

def evaluate_model(y_true, y_pred):
    """
    Step 12: Calculate evaluation metrics
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\nEvaluation Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return precision, recall, f1, cm


# ============================================================================
# STEP 12: VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """
    Step 13a: Plot loss curve during training
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Model Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_error_histogram(train_errors, test_errors, threshold):
    """
    Step 13b: Plot histogram of training and test errors
    """
    plt.figure(figsize=(12, 6))
    
    plt.hist(train_errors, bins=50, alpha=0.7, label='Training Errors', color='blue')
    plt.hist(test_errors, bins=50, alpha=0.7, label='Test Errors', color='orange')
    plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_histogram.png')
    plt.show()


def plot_test_errors_timeline(test_errors, threshold, y_test):
    """
    Step 13c: Plot reconstruction errors over test time
    """
    plt.figure(figsize=(14, 6))
    
    time_steps = np.arange(len(test_errors))
    
    # Plot errors
    plt.plot(time_steps, test_errors, label='Reconstruction Error', color='blue', alpha=0.7)
    plt.axhline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')
    
    # Highlight actual anomalies
    anomaly_indices = np.where(y_test == 1)[0]
    plt.scatter(anomaly_indices, test_errors[anomaly_indices], color='red', s=50, label='Actual Anomalies')
    
    plt.xlabel('Test Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Reconstruction Errors Over Test Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('test_errors_timeline.png')
    plt.show()


def plot_confusion_matrix(cm):
    """
    Plot confusion matrix as heatmap
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("ANOMALY DETECTION IN NUCLEAR POWER PLANT EQUIPMENT")
    print("="*70)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading dataset...")
    df = load_data('SWaT_modified_8_features_dataset.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Step 2: Preprocess
    print("\n[STEP 2] Removing timestamp and normalizing...")
    X, y, scaler = preprocess_data(df)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Step 3: Create sliding windows
    print("\n[STEP 3-4] Creating sliding windows (window_size=50)...")
    X_windows, y_windows = create_sliding_windows(X, y, window_size=50)
    
    # Step 4: Separate normal data
    print("\n[STEP 5-6] Separating normal data for training...")
    X_train, y_train = separate_normal_anomaly(X_windows, y_windows)
    
    # Step 5: Build model
    print("\n[STEP 7] Building Conv1D-LSTM Autoencoder...")
    model = build_autoencoder(window_size=50, n_features=X.shape[1])
    
    # Step 6: Train model
    print("\n[STEP 8] Training the model...")
    history = train_model(model, X_train, epochs=50, batch_size=32)
    
    # Step 7: Calculate training errors for threshold
    print("\n[STEP 9] Calculating reconstruction errors on training data...")
    train_errors = calculate_reconstruction_error(model, X_train)
    
    # Step 8: Determine threshold
    print("\n[STEP 10] Determining anomaly threshold...")
    threshold, mean_error, std_error = determine_threshold(train_errors, threshold_std=3)
    
    # Step 9: Test on all windows
    print("\n[STEP 11] Testing on all windows...")
    test_errors = calculate_reconstruction_error(model, X_windows)
    y_pred = predict_anomalies(test_errors, threshold)
    
    # Step 10: Evaluate
    print("\n[STEP 12] Evaluating model...")
    precision, recall, f1, cm = evaluate_model(y_windows, y_pred)
    
    # Step 11: Visualizations
    print("\n[STEP 13] Creating visualizations...")
    plot_training_history(history)
    plot_error_histogram(train_errors, test_errors, threshold)
    plot_test_errors_timeline(test_errors, threshold, y_windows)
    plot_confusion_matrix(cm)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return model, scaler, threshold, mean_error, std_error


if __name__ == "__main__":
    model, scaler, threshold, mean_error, std_error = main()