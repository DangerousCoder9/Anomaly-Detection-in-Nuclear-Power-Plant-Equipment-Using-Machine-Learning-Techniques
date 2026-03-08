"""
===============================================================================
ANOMALY DETECTION WEB APPLICATION
===============================================================================

OVERVIEW
-------------------------------------------------------------------------------
This Flask application provides a web interface to detect anomalies in
time-series reactor sensor data using a trained Deep Learning Autoencoder.

The system works as follows:

1. A user uploads a CSV file containing reactor sensor readings.
2. The application preprocesses the data and extracts sensor features.
3. Data is normalized using the same scaler used during model training.
4. The data is converted into sliding windows to capture time-series patterns.
5. Each window is passed through a trained Autoencoder model.
6. The reconstruction error is calculated.
7. If the error exceeds a predefined threshold, the system flags it as an anomaly.
8. Results are displayed including anomaly statistics and timestamp-level results.

MODEL
-------------------------------------------------------------------------------
Model Type : Deep Learning Autoencoder
Framework  : TensorFlow / Keras

Training Features (7 Sensors):
    Reactor_Pressure
    Core_Pressure_Differential
    Neutron_Flux
    Reactor_Core_Temperature
    Electrical_Power_Output
    Radiation_Level
    Safety_Monitoring_Parameter

INPUT CSV FORMAT
-------------------------------------------------------------------------------
Required Columns:
    Timestamp
    Reactor_Pressure
    Core_Pressure_Differential
    Neutron_Flux
    Reactor_Core_Temperature
    Electrical_Power_Output
    Radiation_Level
    Safety_Monitoring_Parameter

Optional:
    Label column (Normal / Attack / Anomaly)

TECH STACK
-------------------------------------------------------------------------------
Backend        : Flask
ML Framework   : TensorFlow / Keras
Data Processing: Pandas, NumPy
Evaluation     : Scikit-Learn

MODEL FILES
-------------------------------------------------------------------------------
models/anomaly_detector_model.h5   → Trained Autoencoder
models/scaler.pkl                  → Feature Normalization Scaler
models/threshold.pkl               → Anomaly Detection Threshold
===============================================================================
"""

import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import pickle
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Initialize Flask Application
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Load trained model and preprocessing objects
# -----------------------------------------------------------------------------
print("Loading model...")

# Load trained Autoencoder model
model = load_model('models/anomaly_detector_model.h5')

# Load scaler used during training for feature normalization
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load anomaly detection threshold
with open('models/threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

print("✓ Model loaded successfully!")

# Sliding window size used for time-series processing
WINDOW_SIZE = 50


# -----------------------------------------------------------------------------
# Home Route
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    """
    Displays the main page where users can upload a CSV file
    for anomaly detection.
    """
    return render_template('index.html')


# -----------------------------------------------------------------------------
# Anomaly Detection Route
# -----------------------------------------------------------------------------
@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection function.

    Steps:
    1. Validate uploaded CSV file
    2. Extract sensor data
    3. Normalize features
    4. Create sliding windows
    5. Run model prediction
    6. Compute reconstruction error
    7. Detect anomalies
    8. Generate summary results
    """

    try:

        # ---------------------------------------------------------------------
        # Validate file upload
        # ---------------------------------------------------------------------
        if 'file' not in request.files:
            return render_template('index.html', error='No file uploaded')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected')

        # ---------------------------------------------------------------------
        # Read CSV data into a pandas DataFrame
        # ---------------------------------------------------------------------
        df = pd.read_csv(file)

        # Check if the dataset contains a Label column
        has_label = df.columns[-1].strip().lower() == 'label'

        if has_label:
            # Extract sensor features (skip timestamp and label columns)
            X = df.iloc[:, 1:-1].values

            # Convert text labels into numeric format
            raw_labels = df.iloc[:, -1].astype(str).str.strip().str.lower()

            y = raw_labels.map({
                'normal': 0,
                'attack': 1,
                'anomaly': 1,
                'abnormal': 1
            }).values

        else:
            # Dataset without labels
            X = df.iloc[:, 1:].values
            y = None


        # ---------------------------------------------------------------------
        # Validate feature count
        # ---------------------------------------------------------------------
        expected_features = scaler.n_features_in_

        if X.shape[1] != expected_features:
            return render_template(
                'index.html',
                error=(
                    f'Model expects {expected_features} sensor columns '
                    f'but uploaded file contains {X.shape[1]}.'
                )
            )


        # ---------------------------------------------------------------------
        # Normalize data using the trained scaler
        # ---------------------------------------------------------------------
        X_scaled = scaler.transform(X)


        # ---------------------------------------------------------------------
        # Ensure the file has enough rows for sliding window processing
        # ---------------------------------------------------------------------
        if len(X_scaled) <= WINDOW_SIZE:
            return render_template(
                'index.html',
                error=f'File must contain more than {WINDOW_SIZE} rows.'
            )


        # ---------------------------------------------------------------------
        # Create sliding windows for time-series modeling
        # ---------------------------------------------------------------------
        X_windows = []
        window_indices = []

        for i in range(len(X_scaled) - WINDOW_SIZE):
            X_windows.append(X_scaled[i:i + WINDOW_SIZE])
            window_indices.append(i + WINDOW_SIZE)

        X_windows = np.array(X_windows)


        # ---------------------------------------------------------------------
        # Run model prediction
        # ---------------------------------------------------------------------
        X_pred = model.predict(X_windows, verbose=0)


        # ---------------------------------------------------------------------
        # Calculate reconstruction error
        # ---------------------------------------------------------------------
        errors = np.mean(np.square(X_windows - X_pred), axis=(1, 2))


        # Detect anomalies using the threshold
        predictions = (errors > threshold).astype(int)


        # ---------------------------------------------------------------------
        # Prepare row-level results for visualization
        # ---------------------------------------------------------------------
        timestamps = df.iloc[:, 0].values

        row_results = []

        for idx, (wi, err, pred) in enumerate(zip(window_indices, errors, predictions)):
            row_results.append({
                'timestamp': timestamps[wi],
                'error': round(float(err), 6),
                'label': 'Anomaly' if pred == 1 else 'Normal'
            })


        # ---------------------------------------------------------------------
        # Summary statistics
        # ---------------------------------------------------------------------
        results = {
            'total_windows': int(len(predictions)),
            'anomalies_detected': int(np.sum(predictions)),
            'normal_detected': int(len(predictions) - np.sum(predictions)),
            'anomaly_percentage': round(float(np.sum(predictions) / len(predictions) * 100), 2),
            'threshold': round(float(threshold), 6),
            'mean_error': round(float(np.mean(errors)), 6),
            'max_error': round(float(np.max(errors)), 6),
            'min_error': round(float(np.min(errors)), 6),
            'row_results': row_results,
            'has_metrics': False
        }


        # ---------------------------------------------------------------------
        # If labels are available, compute evaluation metrics
        # ---------------------------------------------------------------------
        if y is not None:

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix
            )

            y_windows = np.array([
                y[i + WINDOW_SIZE] for i in range(len(y) - WINDOW_SIZE)
            ])

            if len(y_windows) == len(predictions):

                cm = confusion_matrix(y_windows, predictions)

                results['has_metrics'] = True

                results['metrics'] = {
                    'accuracy': round(float(accuracy_score(y_windows, predictions)), 4),
                    'precision': round(float(precision_score(y_windows, predictions, zero_division=0)), 4),
                    'recall': round(float(recall_score(y_windows, predictions, zero_division=0)), 4),
                    'f1_score': round(float(f1_score(y_windows, predictions, zero_division=0)), 4),
                    'tn': int(cm[0, 0]),
                    'fp': int(cm[0, 1]),
                    'fn': int(cm[1, 0]),
                    'tp': int(cm[1, 1])
                }


        # Render result page
        return render_template('result.html', results=results)


    except Exception as e:
        # Handle unexpected errors
        return render_template('index.html', error=str(e))


# -----------------------------------------------------------------------------
# Run Flask Application
# -----------------------------------------------------------------------------
if __name__ == '__main__':

    # debug=True enables automatic reload during development
    # host='0.0.0.0' allows access from other devices in the network

    app.run(debug=True, host='0.0.0.0', port=5000)