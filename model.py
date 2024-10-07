from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize the FastAPI app
app = FastAPI()

# Load the trained models (from Task 4)
risk_model = joblib.load("/home/semre/credit_risk_analysis/models/risk_model.pkl")  # Risk prediction model
credit_model = joblib.load("/home/semre/credit_risk_analysis/models/credit_model.pkl")  # Credit score model
loan_model = joblib.load("/home/semre/credit_risk_analysis/models/loan_model.pkl")  # Loan amount and duration model

# Input data models
class CustomerData(BaseModel):
    transaction_history: list
    age: int
    income: float
    # Add other relevant customer features needed by your model

class RiskData(BaseModel):
    risk_probability: float

# Endpoint 1: Predict risk probability
@app.post("/predict-risk")
async def predict_risk(data: CustomerData):
    # Preprocess the input data (scaling, encoding, etc.)
    input_data = np.array(data.transaction_history).reshape(1, -1)
    risk_probability = risk_model.predict_proba(input_data)[0][1]
    
    return {"risk_probability": risk_probability}

# Endpoint 2: Predict credit score based on risk probability
@app.post("/predict-credit-score")
async def predict_credit_score(data: RiskData):
    credit_score = credit_model.predict([[data.risk_probability]])[0]
    
    return {"credit_score": credit_score}

# Endpoint 3: Predict optimal loan amount and duration
@app.post("/predict-loan")
async def predict_loan(data: CustomerData):
    # Preprocess the input data (scaling, encoding, etc.)
    input_data = np.array(data.transaction_history).reshape(1, -1)
    
    # Predict loan amount and duration (multi-output regression)
    loan_amount, loan_duration = loan_model.predict(input_data)[0]
    
    return {
        "loan_amount": loan_amount,
        "loan_duration": loan_duration
    }
