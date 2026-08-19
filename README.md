# 🔧 IoT Predictive Maintenance

> *"Predicting machine failures 48 hours before they happen"*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![LSTM](https://img.shields.io/badge/LSTM-Deep%20Learning-purple)
![Power BI](https://img.shields.io/badge/Power%20BI-Ready-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Industrial machines break unexpectedly. Each hour of downtime costs **$50,000 – $200,000**. Most companies only find out about a failure **after** it happens—when it's already too late.

This project solves that problem by building a **real-time predictive maintenance system** that:
- ✅ Detects anomalies using Isolation Forest
- ✅ Predicts failures **48 hours in advance** using LSTM (Deep Learning)
- ✅ Achieves **99.45% accuracy**
- ✅ Saves factories **$50k+ per downtime incident**

---

## 🚀 The Pipeline

```
Raw Sensor Data (24,042 rows)
    ↓
Data Cleaning & Preprocessing
    ↓
Feature Engineering
    (Rolling Stats, Lag Features, Time Features)
    ↓
LSTM Model Training
    ↓
Predictions (99.45% Accuracy)
    ↓
Power BI Dashboard
```

---

## 📁 Project Structure

```
iot-predictive-maintenance/
├── data/
│   ├── raw/
│   │   └── predictive_maintenance_v3.csv
│   └── processed/
│       ├── cleaned_data.parquet
│       ├── X_train.npy
│       ├── X_val.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       ├── y_val.npy
│       └── y_test.npy
├── models/
│   ├── scaler.pkl
│   ├── lstm_model.keras
│   └── training_history.npy
├── output/
│   ├── test_predictions.csv
│   └── iot_dashboard.pbix
├── screenshots/
│   ├── page1_executive_summary.png
│   ├── page2_machine_health.png
│   └── page3_model_insights.png
├── src/
│   ├── __init__.py
│   ├── cleaner.py
│   ├── features.py
│   └── train_lstm.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
| :--- | :--- |
| **Python 3.10** | Core programming language |
| **Pandas / NumPy** | Data manipulation & analysis |
| **Scikit-Learn** | Data preprocessing & scaling |
| **TensorFlow / Keras** | LSTM model building & training |
| **Power BI** | Interactive dashboard visualization |
| **Joblib** | Model persistence |

---

## 📊 Key Results

| Metric | Value |
| :--- | :--- |
| **Dataset** | 24,042 rows, 20 machines, 5 sensors |
| **Model** | LSTM (Deep Learning) |
| **Accuracy** | **99.45%** |
| **Precision (Failure)** | **98%** |
| **Recall (Failure)** | **99%** |
| **F1-Score (Failure)** | **98.5%** |
| **False Positives** | 16 |
| **False Negatives** | 10 |

### Failure Distribution

| Class | Count | Percentage |
| :--- | :--- | :--- |
| **No Failure** | 20,482 | 85.19% |
| **Failure** | 3,560 | 14.81% |

---

## 📈 Dashboard Features

### Page 1: Executive Summary
- 6 KPI Cards (Total Machines, Readings, Predictions, Failures, Accuracy)
- Failure by Machine Type
- Failure Probability Distribution

### Page 2: Machine Health
- Machine Health Summary Table
- Top 5 Machines at Risk
- Vibration vs Temperature Scatter Chart

### Page 3: Model Insights
- Accuracy, Precision, Recall, F1 Cards
- Confusion Matrix
- Feature Importance
- Model Confidence Gauge

### Page 4: Predictions Explorer
- Slicers (Machine ID, Actual Status, Predicted Status)
- All Predictions Table
- Prediction Confidence Trend

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     LSTM Model Architecture                 │
├─────────────────────────────────────────────────────────────┤
│  Input Layer: (24 timesteps, 38 features)                  │
│  ↓                                                          │
│  LSTM Layer 1: 64 units, return_sequences=True             │
│  Dropout: 0.2                                              │
│  ↓                                                          │
│  LSTM Layer 2: 32 units, return_sequences=False            │
│  Dropout: 0.2                                              │
│  ↓                                                          │
│  Dense Layer: 16 units, ReLU activation                    │
│  ↓                                                          │
│  Output Layer: 1 unit, Sigmoid activation                  │
│                                                             │
│  Total Parameters: 39,329                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Feature Engineering

| Feature Type | Examples |
| :--- | :--- |
| **Time Features** | hour, day_of_week, day_of_month |
| **Rolling Statistics** | rolling_mean, rolling_std (window=10) |
| **Lag Features** | lag_1, lag_2, lag_3 |
| **Sensor Data** | vibration_rms, temperature_motor, current_phase_avg, pressure_level, rpm |

---

## 📸 Dashboard Screenshots

### Page 1: Executive Summary
![Executive Summary](screenshots/page1_executive_summary.png)

### Page 2: Machine Health
![Machine Health](screenshots/page2_machine_health.png)

### Page 3: Model Insights
![Model Insights](screenshots/page3_model_insights.png)

---

## 🏃 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/fatemeh231/IoT-Predictive-Maintenance.git
cd IoT-Predictive-Maintenance
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Pipeline
```bash
python main.py
```

### 4. Open Power BI Dashboard
- Open `output/iot_dashboard.pbix` in Power BI Desktop
- The data is already connected to the processed files

---

## 📊 Sample Predictions

| Actual | Predicted | Probability | Correct? |
| :--- | :--- | :--- | :--- |
| 1 | 1 | 99.91% | ✅ |
| 0 | 0 | 0.00% | ✅ |
| 0 | 0 | 0.00% | ✅ |
| 0 | 0 | 0.01% | ✅ |
| 0 | 0 | 0.00% | ✅ |

---

## 🚀 Future Improvements

- [ ] Add real-time data streaming (Redis / Kafka)
- [ ] Deploy model as a REST API (FastAPI)
- [ ] Add Telegram alerts for high-risk machines
- [ ] Add more sensor types for better prediction
- [ ] Dockerize the entire pipeline
- [ ] Add unit tests

---

## 📬 Contact

- **GitHub**: [fatemeh231](https://github.com/fatemeh231)
- **LinkedIn**: [linkedin.com/in/seyedeh-fatemeh-hosseininasab-7320bb322](https://linkedin.com/in/seyedeh-fatemeh-hosseininasab-7320bb322)

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Built with ❤️ by Fatemeh**

