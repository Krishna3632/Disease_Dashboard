from flask import Flask, render_template, request,jsonify
import requests
import numpy
import joblib
from io import BytesIO
import sklearn
import numpy as np

app = Flask(__name__)

@app.route('/pred', methods=['POST'])
def predict_diabetes():
    # Get JSON data from the request
    dib=joblib.load("Diabetes_Pred.pkl")
    data = request.get_json()

    # Extracting the required fields
    age = data.get('age')
    bmi = data.get('bmi')
    glucose = data.get('glucose')
    bloodpressure = data.get('bloodpressure')

    # Check for missing data
    if age is None or bmi is None or glucose is None or bloodpressure is None:
        return jsonify({'error': 'Missing data'}), 400

    # Validate and convert input data
    try:
        age = float(age)
        bmi = float(bmi)
        glucose = float(glucose)
        bloodpressure = float(bloodpressure)
    except ValueError:
        return jsonify({'error': 'Invalid data type, expected numbers'}), 400

    # Prepare input data for prediction
    input_data = np.array([[age, bmi, glucose, bloodpressure]])

    # Make prediction
    try:
        prediction = dib.predict(input_data)[0]  # Assuming dib is your model
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Return the result
    result = {
        'Result': str(prediction)
    }
    
    return jsonify(result), 201

if __name__ == "__main__":
    app.run(debug=True)