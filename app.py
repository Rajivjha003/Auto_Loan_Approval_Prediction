import joblib
from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
random_forest_loan = joblib.load(r'E:\Predicting Auto Loan Approval\random_forest_loan.pkl')

scaler = joblib.load('E:\Predicting Auto Loan Approval\scaling_chd.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json  # Access the JSON data directly
    features = [
        data['CreditType'],
        data['Job Hours'],
        data['Income'],
        data['LTV'],
        data['Term'],
        data['Price'],
        data['Downpayment'],
        data['BookValue'],
        data['Amount Financed'],
        data['APR'],
        data['Monthly Payment'],
        data['Monthly Dedt'],
        data['No. of Tradelines'],
        data['FICO'],
        data['ClearFraud Score'],
        data['Vehicle Year'],
        data['Vehicle Age'],
        data['Vehicle Miles'],
        data['DTI'],
        data['LTV_DP']
    ]
    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = random_forest_loan.predict(scaled_features)[0]
    return jsonify(prediction)


@app.route('/predict',methods=['POST'])
def predict():
    data=[int(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    output=random_forest_loan.predict(final_input)[0]
    
    if output == 1:
        prediction_text = "approved"
    else:
        prediction_text = "rejected"

    return render_template("home.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
