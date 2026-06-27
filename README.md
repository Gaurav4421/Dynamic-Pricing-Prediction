# 🚖 Uber & Lyft Dynamic Pricing Predictor

> **End-to-End Machine Learning Portfolio Project**
> Predict Uber & Lyft ride fares using historical trip data from **Boston, MA** with an optimized **XGBoost** regression model and an interactive **Streamlit** web application.

---

## 🌐 Live Demo

**🚀 Try the App:**
https://dynamic-pricing-prediction-kngm5frdq8rcwzse2qiyry.streamlit.app/

---

## 📂 GitHub Repository

https://github.com/Gaurav4421/Dynamic-Pricing-Prediction

---

# 📌 Project Overview

Ride-hailing platforms such as **Uber** and **Lyft** use dynamic pricing that changes according to multiple factors including:

* Distance
* Ride type
* Pickup & drop-off location
* Time of day
* Day of week
* Surge pricing

This project builds a complete machine learning pipeline that predicts ride fares using historical ride data from Boston.

The project covers the entire ML lifecycle:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Hyperparameter tuning
* Model serialization
* Interactive deployment using Streamlit

---

# 📦 Dataset

| Property        | Details                  |
| --------------- | ------------------------ |
| Dataset         | Uber & Lyft Ride Dataset |
| Source          | Kaggle                   |
| Location        | Boston, Massachusetts    |
| Size            | 693,000+ rides           |
| Raw Features    | 57                       |
| Target Variable | `price`                  |

### Features Used

* cab_type
* ride_name
* distance
* surge_multiplier
* pickup_location
* destination
* hour
* day_of_week
* is_rush_hour
* is_weekend

> **Note:** Predictions are generated in **USD ($)** because the model was trained exclusively on ride data collected in Boston, MA.

---

# 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

# 📁 Project Structure

```text
Dynamic-Pricing-Prediction/
│
├── app.py
├── train.py
├── predict.py
├── preprocessing.py
├── feature_engineering.py
├── evaluate.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── best_model.pkl
│   ├── feature_names.pkl
│   └── label_encoders.pkl
│
└── data/
    └── rideshare_kaggle.csv
```

---

# 🚀 Installation

## Clone the repository

```bash
git clone https://github.com/Gaurav4421/Dynamic-Pricing-Prediction.git

cd Dynamic-Pricing-Prediction
```

---

## Create a virtual environment

```bash
python -m venv venv
```

Activate it

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Download Dataset

Download the **Uber & Lyft Boston Dataset** from Kaggle and place

```text
rideshare_kaggle.csv
```

inside the

```text
data/
```

folder.

---

# ▶️ Train the Model

```bash
python train.py
```

Training automatically performs:

* Data preprocessing
* Feature engineering
* Multiple model training
* Hyperparameter tuning
* Best model selection
* Model serialization

Saved artifacts:

```
model/
├── best_model.pkl
├── feature_names.pkl
└── label_encoders.pkl
```

---

# ▶️ Launch the Application

```bash
streamlit run app.py
```

---

# 🔄 Machine Learning Workflow

```text
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train Multiple Models
      │
      ▼
Model Evaluation
      │
      ▼
RandomizedSearchCV
      │
      ▼
Best Model Saved
      │
      ▼
Streamlit Deployment
```

---

# 🤖 Models Compared

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost
* Tuned XGBoost (RandomizedSearchCV)

---

# 📈 Model Performance

| Model             |      MAE |     RMSE |       R² |
| ----------------- | -------: | -------: | -------: |
| Linear Regression |     5.08 |     6.31 |     0.52 |
| Decision Tree     |     1.10 |     1.65 |     0.97 |
| Random Forest     |     1.12 |     1.75 |     0.96 |
| XGBoost           |     1.08 |     1.61 |     0.97 |
| ⭐ Tuned XGBoost   | **1.02** | **1.55** | **0.97** |

---

# 💡 Key Highlights

* End-to-end Machine Learning project
* 693K real-world ride records
* Clean modular project structure
* Feature engineering pipeline
* Compared four regression algorithms
* Hyperparameter tuning using RandomizedSearchCV
* Saved reusable model artifacts with Joblib
* Interactive Streamlit web application
* Fully deployed to Streamlit Community Cloud

---

# 🔮 Future Improvements

* Weather-aware fare prediction
* Traffic congestion features
* Route-aware distance calculation
* One-Hot Encoding comparison
* Continuous model retraining
* API deployment using FastAPI

---

# 👨‍💻 Author

**Gaurav Verma**

🔗 GitHub
https://github.com/Gaurav4421

🔗 LinkedIn
https://www.linkedin.com/in/YOUR-LINKEDIN-URL

---

# 📄 License

This project is licensed under the **MIT License**.
