# 🏥 Healthcare Claims Analyzer

> *"Predicting claim denials before they happen"*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange)
![Power BI](https://img.shields.io/badge/Power%20BI-Ready-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Hospitals lose millions annually to claim denials. This project builds an **end-to-end analytics pipeline** that:

- ✅ Processes **18+ million claims** from Medicare data
- ✅ Cleans and joins **Inpatient, Outpatient, and Carrier** claims
- ✅ Builds a **Machine Learning model** to predict claim denials
- ✅ Achieves **97.13% accuracy** and **96% recall** for denied claims
- ✅ Flags **high-risk claims** before submission
- ✅ Visualizes insights in an **interactive Power BI dashboard**

---

## 🚀 The Pipeline

```
Raw Data (Beneficiary + Claims)
    ↓
ETL Pipeline (Python/Pandas)
    ↓
Data Cleaning & Joining
    ↓
Feature Engineering (Age, Diagnosis, Claim Type)
    ↓
Machine Learning (Random Forest)
    ↓
Predictions (DENIAL_PROBABILITY, PREDICTED_DENIED)
    ↓
Power BI Dashboard (Executive Summary + Model Insights)
```

---

## 📁 Project Structure

```
healthcare-claims-analyzer/
├── data/
│   └── processed/
│       └── claims_master.parquet
├── models/
│   └── claim_denial_model.pkl
├── notebooks/
│   └── 01_data_exploration.ipynb
├── output/
│   └── claims_dashboard.pbix
├── screenshots/
│   ├── page1_executive_summary.png
│   └── page2_model_insights.png
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── cleaner.py
│   └── model.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

| Tool | Purpose |
| :--- | :--- |
| **Python 3.10** | Core programming language |
| **Pandas** | Data manipulation & cleaning |
| **NumPy** | Numerical operations |
| **Scikit-Learn** | Machine Learning (Random Forest) |
| **Imbalanced-Learn** | SMOTE for class imbalance |
| **PyArrow** | Parquet file support |
| **Power BI** | Dashboard & visualization |

---

## 📊 Key Results

| Metric | Value |
| :--- | :--- |
| **Total Claims** | 18,231,617 |
| **Denied Claims** | 417,636 (2.29%) |
| **Model Accuracy** | **97.13%** |
| **Recall (Denied)** | **96%** |
| **Precision (Denied)** | 7% |
| **F1-Score (Denied)** | 14% |
| **High-Risk Claims (>50%)** | 5,465,051 (29.98%) |

### Claim Type Distribution

| Claim Type | Total Claims | Denied Claims | Denial Rate |
| :--- | :--- | :--- | :--- |
| Carrier | 1,121,004 | 417,636 | **37.26%** |
| Outpatient | 575,092 | 0 | 0.00% |
| Inpatient | 58,066 | 0 | 0.00% |

---

## 📈 Feature Importance

| Feature | Importance |
| :--- | :--- |
| **CLAIM_TYPE** | 25.7% |
| **PTNT_DSCHRG_STUS_CD** | 16.6% |
| **AGE** | 16.4% |
| **PRNCPAL_DGNS_CD** | 15.5% |
| **CLM_TOT_CHRG_AMT** | 8.6% |

---

## 🏃 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/fatemeh231/healthcare-claims-analyzer.git
cd healthcare-claims-analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Your Data
Place your Medicare data files in `data/raw/`

### 4. Run the Pipeline
```bash
python main.py
```

### 5. Open Power BI Dashboard
- Open `output/claims_dashboard.pbix`
- Connect to `data/processed/claims_master.parquet`

---

## 📊 Dashboard 


## 🧠 Business Insights

| Insight | Action |
| :--- | :--- |
| **Carrier claims have 37.26% denial rate** | Investigate Carrier billing practices |
| **Inpatient/Outpatient claims have 0% denial** | Maintain current practices |
| **Diagnosis codes are top predictor** | Review coding accuracy |
| **Age impacts denial probability** | Targeted review for high-risk age groups |

---

## 🚀 Future Improvements

- [ ] Add real-time prediction API
- [ ] Integrate with live claims systems
- [ ] Add explainability (SHAP/LIME)
- [ ] Deploy to cloud (AWS/Azure)

---

## 📬 Contact

- **LinkedIn**: https://www.linkedin.com/in/seyedeh-fatemeh-hosseininasab-7320bb322/

---

## 📄 License

This project is licensed under the **MIT License**.

---

**Built with ❤️ by Fatemeh**
