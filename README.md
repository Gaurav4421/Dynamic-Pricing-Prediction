# 🚗 Dynamic Pricing Prediction

A machine learning project that predicts Uber and Lyft ride fares using historical ride data from Boston, MA. The project covers the complete workflow from data preprocessing and feature engineering to model training, evaluation, and deployment with Streamlit.

---

## 📌 Problem Statement

Ride prices on platforms like Uber and Lyft change dynamically based on factors such as distance, ride type, surge pricing, and time of day.

The objective of this project is to build a regression model that can predict ride fares using these features.

---

## 📦 Dataset

| Property | Details                          |
| -------- | -------------------------------- |
| Dataset  | Uber & Lyft Dataset – Boston, MA |
| Source   | Kaggle                           |
| Size     | ~693,000 rows × 57 columns       |
| Target   | `price`                          |

### Features Used

* `cab_type`
* `name`
* `distance`
* `surge_multiplier`
* `source`
* `destination`
* `hour`
* `day_of_week`
* `is_rush_hour`
* `is_weekend`

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

## 📁 Project Structure

```text
Dynamic-Pricing-Prediction/
│
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate.py
│   ├── train.py
│   └── predict.py
│
├── model/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

Clone the repository

```bash
git clone https://github.com/yourusername/Dynamic-Pricing-Prediction.git
cd Dynamic-Pricing-Prediction
```

Create a virtual environment

```bash
python -m venv venv
```

Install dependencies

```bash
pip install -r requirements.txt
```

Download the Kaggle dataset and place `rideshare_kaggle.csv` inside the `data/` folder.

---

## ▶️ Run the Project

### Run the notebook

```bash
jupyter notebook notebooks/dynamic_pricing.ipynb
```

### Train the models

```bash
python src/train.py
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

---

## 🔄 Workflow

```text
Raw Data
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
Hyperparameter Tuning
    │
    ▼
Save Best Model
    │
    ▼
Streamlit App
```

---

## 📊 Models Trained

* Linear Regression
* Decision Tree
* Random Forest
* XGBoost

The best-performing model is further tuned using **RandomizedSearchCV** before being saved.

---

## 📈 Results

| Model | MAE | RMSE | R² |
|------|----:|-----:|---:|
| Linear Regression | 5.08 | 6.31 | 0.52 |
| Decision Tree | 1.10 | 1.65 | 0.97 |
| Random Forest | 1.12 | 1.75 | 0.96 |
| XGBoost | 1.08 | 1.61 | 0.97 |
| **XGBoost (Tuned)** | **1.02** | **1.55** | **0.97** |

---

## 💡 Key Takeaways

* Compared four regression models.
* XGBoost achieved the best overall performance.
* Engineered time-based features improved prediction quality.
* Built a reusable preprocessing and training pipeline.
* Deployed the trained model using Streamlit.

---

## 🔮 Future Improvements

* Include weather-related features.
* Experiment with One-Hot Encoding.
* Add traffic and route information.
* Deploy the application online.

---

## 👤 Author

Gaurav Verma

* GitHub: [https://github.com/Gaurav4421](https://github.com/Gaurav4421)
* LinkedIn: [https://linkedin.com/in/your-linkedin](https://www.linkedin.com/in/gaurav-verma-00a524342/)

---

## 📄 License

This project is licensed under the MIT License.
