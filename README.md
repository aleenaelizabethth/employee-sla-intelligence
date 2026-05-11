# 📊 Employee SLA Intelligence Dashboard

> Predictive ML model to forecast SLA adherence and workforce performance — built with Python, Scikit-learn, and Streamlit.

---

## 🔍 Project Overview

This project analyses employee performance data to **predict whether SLA (Service Level Agreement) targets will be met or breached**. It combines exploratory data analysis, machine learning classification, and an interactive Streamlit dashboard for real-time predictions.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## 📁 Project Structure

```
employee-sla-intelligence/
│
├── employee_project.ipynb      # EDA + ML model building
├── employee_sla_app.py         # Streamlit dashboard app
└── employee_sla_status.csv     # Dataset
```

---

## 🔄 Workflow

### 1. Exploratory Data Analysis
- Analysed employee distribution across **Teams and Shifts**
- Visualised **SLA Status distribution** using bar charts and count plots
- Detected **outliers** using box plots
- Built **correlation heatmap** to identify key performance drivers

### 2. Data Preprocessing
- Handled missing values (mean/median imputation)
- Encoded categorical features (Team, Shift) using One-Hot Encoding
- Scaled features using **MinMaxScaler**
- Mapped target variable: `Met → 1`, `Breached → 0`

### 3. Machine Learning Model
- Algorithm: **K-Nearest Neighbours (KNN) Classifier**
- Tuned hyperparameters using **GridSearchCV** (10-fold cross-validation)
- Best parameters: `n_neighbors=9`, `weights='distance'`

### 4. Streamlit Dashboard
- Interactive app for **real-time SLA prediction**
- Input employee metrics → get instant Met/Breached prediction

---

## ▶️ How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn streamlit

# Run the Streamlit app
streamlit run employee_sla_app.py
```

---

## 👩‍💻 Author

**Aleena Elizabeth Thomas**  
[LinkedIn](https://linkedin.com/in/aleena-elizabeth-thomas) · [GitHub](https://github.com/aleenaelizabethth)
