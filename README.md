# Anomaly Detection in Nuclear Power Plant Equipment Using Machine Learning

This project focuses on detecting abnormal behavior in nuclear power plant equipment using machine learning techniques.

Nuclear power plants rely on multiple sensors to continuously monitor critical parameters such as:

* Temperature
* Pressure
* Coolant Flow Rate
* Turbine Speed
* Radiation Levels

Identifying unusual patterns in this sensor data is essential to ensure **safety, reliability, and efficient plant operation**.

The goal of this project is to build an **intelligent anomaly detection system** that learns normal operational patterns from historical sensor data and detects deviations that may indicate **equipment malfunction or unsafe conditions**.

---

# Project Overview

The system analyzes **time-series data** collected from different sensors in the nuclear power plant.

Each sensor records measurements at regular time intervals. These readings are processed and used to train a **deep learning model capable of identifying abnormal behavior**.

Instead of relying on predefined thresholds, the model **learns normal system behavior** and detects anomalies using **reconstruction error**.

---

# Key Features

* Detection of abnormal patterns in nuclear plant sensor data
* Time-series analysis using deep learning
* Automatic learning of normal equipment behavior
* Early detection of potential equipment faults
* Visualization of anomaly detection results

---

# Dataset Structure

The dataset contains **time-stamped sensor readings** from different components of the nuclear power plant.

| Timestamp | Temperature | Pressure | Flow Rate | Turbine Speed | Radiation | Label            |
| --------- | ----------- | -------- | --------- | ------------- | --------- | ---------------- |
| Time      | Sensor1     | Sensor2  | Sensor3   | Sensor4       | Sensor5   | Normal / Anomaly |

Each row represents a **sensor reading at a specific moment in time**.

---

# Data Preprocessing

Several preprocessing steps are applied before training the model:

1. Remove the **timestamp column**
2. Normalize sensor values to a common scale
3. Convert sequential data into **time windows using a sliding window technique**
4. Split the data into **training and testing datasets**

The **sliding window technique** helps the model learn temporal relationships in the sensor data.

---

# Model Architecture

The anomaly detection system uses a **Conv1D–LSTM Autoencoder architecture**.

### Components

* **Conv1D Layer**
  Detects short-term temporal patterns in the sensor data.

* **LSTM Encoder**
  Compresses the time-series sequence into a compact latent representation.

* **Repeat Vector**
  Prepares the encoded representation for reconstruction.

* **LSTM Decoder**
  Reconstructs the original input sequence.

* **Dense Layer**
  Outputs reconstructed sensor values.

The model learns to reconstruct **normal operational patterns**. When abnormal sequences are given as input, the **reconstruction error increases**, indicating a potential anomaly.

---

# Anomaly Detection Method

The anomaly detection process works as follows:

1. The model reconstructs the input sequence.
2. The **reconstruction error** is calculated.
3. A threshold is determined using the **mean and standard deviation of training errors**.

If the reconstruction error exceeds the threshold, the sequence is classified as an **anomaly**.

---

# Evaluation Metrics

Model performance is evaluated using:

* Precision
* Recall
* F1-Score
* Confusion Matrix

These metrics help measure how effectively the system detects anomalies while minimizing false alarms.

---

# Applications

This system can be applied in several domains:

* Nuclear power plant monitoring
* Industrial equipment monitoring
* Predictive maintenance
* Fault detection in complex machinery
* Safety monitoring systems

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---
