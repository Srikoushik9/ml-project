\# Credit Card Fraud Detection – Machine Learning Project



\##  Overview

This project implements an end-to-end Credit Card Fraud Detection system designed to identify fraudulent transactions from highly imbalanced transactional data. The solution covers data analysis, feature engineering, model training, evaluation, API development, and cloud deployment for real-time inference.



---



\##  Problem Statement

Credit card fraud detection is a challenging classification problem due to extreme class imbalance and evolving fraud patterns. The goal of this project is to maximize fraud detection recall while minimizing false positives in real-world transaction data.



---



\##  Approach

\- Performed exploratory data analysis to understand fraud patterns and class imbalance

\- Applied feature scaling and resampling techniques (SMOTE) to improve minority class representation

\- Trained and evaluated multiple machine learning models

\- Optimized model performance using hyperparameter tuning

\- Exposed predictions via a REST API for real-time usage

\- Deployed the trained model on Azure ML for scalable inference



---



\## 🤖 Models Used

\- Logistic Regression  

\- Random Forest  

\- XGBoost  



Model selection was based on performance metrics such as \*\*Recall\*\*, \*\*Precision\*\*, \*\*F1-score\*\*, and \*\*ROC-AUC\*\*.



---



\## ⚙️ Tech Stack

\- \*\*Programming Language:\*\* Python  

\- \*\*Libraries:\*\* NumPy, Pandas, Scikit-learn, XGBoost, Imbalanced-learn  

\- \*\*API:\*\* FastAPI  

\- \*\*Model Optimization:\*\* GridSearchCV, RandomizedSearchCV  

\- \*\*Deployment:\*\* Azure Machine Learning  

\- \*\*Visualization:\*\* Matplotlib, Seaborn  



---



\##  Key Features

\- Handles highly imbalanced datasets using SMOTE

\- Optimized fraud detection with recall-focused metrics

\- FastAPI-based REST endpoint for real-time predictions

\- Cloud deployment using Azure ML for scalability

\- Admin dashboard with analytics and reporting to support data-driven decisions



---



\##  How to Run Locally



```bash

\# Clone the repository

git clone https://github.com/<your-username>/credit-card-fraud-detection.git



\# Install dependencies

pip install -r requirements.txt



\# Run the FastAPI app

uvicorn app.main:app --reload



