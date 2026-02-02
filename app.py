from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np


with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)


class Transaction(BaseModel):
    transaction_id: int
    amount: float
    transaction_hour: int
    merchant_category: int
    foreign_transaction: int
    location_mismatch: int
    device_trust_score: float
    velocity_last_24h: int
    cardholder_age: int
    night_transaction: int
    high_amount: int


    
app = FastAPI()


@app.post("/predict")
def predict(transaction: Transaction):
    
    data = np.array([[  
        transaction.transaction_id,
        transaction.amount,
        transaction.transaction_hour,
        transaction.merchant_category,
        transaction.foreign_transaction,
        transaction.location_mismatch,
        transaction.device_trust_score,
        transaction.velocity_last_24h,
        transaction.cardholder_age,
        transaction.night_transaction,
        transaction.high_amount
    ]])


    
    prediction = model.predict(data)
    probability = model.predict_proba(data)[:, 1]  

    return {
        "is_fraud": int(prediction[0]),
        "fraud_probability": float(probability[0])
    }
