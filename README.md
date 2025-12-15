📉 Customer Churn Prediction using Machine Learning & Streamlit

This project predicts whether a customer is likely to churn (leave the service) based on their subscription details, usage behavior, and billing information.
A machine learning model is trained using historical data and deployed as a Streamlit web application for real-time predictions.

🔍 Project Overview

Customer churn is a major problem for subscription-based businesses.
The goal of this project is to help businesses identify customers who are at risk of leaving so that preventive actions can be taken in advance.

The project includes:

Data preprocessing and feature engineering

Machine learning model training

Model evaluation

A user-friendly Streamlit web interface

Project Structure
customer_churn_project/
│
├── train_model.py                  
├── app.py                          
├── customer_churn_large_dataset.xlsx
├── customer_churn_model.pkl        
├── scaler.pkl                      
├── requirements.txt
└── README.md

📊 Dataset Description

The dataset contains customer information such as:

Age

Gender

Location

Subscription length (months)

Monthly bill amount

Total data usage (GB)

Churn (Target variable: 0 = No, 1 = Yes)

Unnecessary columns like CustomerID and Name are removed during preprocessing.

⚙️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

🚀 How to Run the Project
1️⃣ Clone or Download the Project

Place all files inside one folder and open it in VS Code.

2️⃣ Create and Activate Virtual Environment
python -m venv venv


Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (Run Once)
python train_model.py


This will generate:

customer_churn_model.pkl

scaler.pkl

5️⃣ Run the Streamlit App
streamlit run app.py


Open the displayed URL in your browser to use the application.

🖥️ Application Features

User-friendly input sliders and dropdowns

Real-time churn prediction

Probability-based output

Clean and simple UI

📈 Model Used

Random Forest Classifier

Feature scaling using StandardScaler

Achieves good accuracy on large datasets
