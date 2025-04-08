from flask import Blueprint, render_template, request
from app.model_utils import load_lstm_model, predict_rul
import pandas as pd

main = Blueprint('main', __name__)
model = load_lstm_model()  # loads lstm_model.keras from models/

@main.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            prediction = predict_rul(model, df)
    return render_template('index.html', prediction=prediction)
