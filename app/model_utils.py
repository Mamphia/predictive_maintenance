import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from app.preprocess import preprocess_input

def load_lstm_model(path='models/lstm_model.keras'):
    return load_model(path)

def predict_rul(model, df):
    processed = preprocess_input(df)
    prediction = model.predict(processed)
    return prediction.flatten().tolist()
