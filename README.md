# 💼 Income Prediction ML Project

## 📊 Problem Statement
Predict whether a person earns more than 50K per year based on demographic and work-related features using machine learning models.

## 📌 Project Overview
This project is an end-to-end Machine Learning pipeline that predicts whether a person's income is greater than 50K or not using the Adult Census dataset.

---

## 📁 Dataset
The project uses the Adult Census dataset stored in:
`data/raw/adult.csv`

## 🚀 Features
- Data Ingestion from online source  
- Data Preprocessing & Cleaning  
- Feature Engineering  
- Training multiple ML models  
- Model Evaluation & Selection  
- Deployment using Streamlit  

---

## 🧠 Models Used
- Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- Support Vector Machine (SVM)  
- Naive Bayes  
- K-Nearest Neighbors (KNN)  

---

## 📂 Project Structure

```bash
project_income/
│── data/
│── src/
│── pipelines/
│── models/
│── artifacts/
│── deployment/
│── main.py
│── requirements.txt

## ⚙️ Installation

git clone https://github.com/your-username/project_income.git
cd project_income
python -m venv .venv_income
.venv_income\Scripts\activate
pip install -r requirements.txt

## ▶️ Run Project

python main.py

## 🌐 Run Web App

streamlit run deployment/app.py

## 📈 Results
- Trained 7 different ML models  
- Automatically selected best model based on F1-score  
- Achieved strong performance on test dataset  