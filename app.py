import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Resolve absolute path to the pickle file on Vercel
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')

# Global variable to hold loaded model
model = None
model_load_error = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model_load_error = f"Model file not found at path: {MODEL_PATH}"
except Exception as e:
    model_load_error = f"Failed to load model: {str(e)}"


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': model_load_error or 'Model is not loaded.'
        }), 500

    try:
        # Extract features from POST payload safely
        education = float(request.form.get('Education', 0))
        joining_year = float(request.form.get('JoiningYear', 2020))
        city = float(request.form.get('City', 0))
        payment_tier = float(request.form.get('PaymentTier', 1))
        age = float(request.form.get('Age', 25))
        gender = float(request.form.get('Gender', 0))
        ever_benched = float(request.form.get('EverBenched', 0))
        experience = float(request.form.get('ExperienceInCurrentDomain', 0))

        # Array shape must match the 8 trained model features
        features = np.array([[
            education,
            joining_year,
            city,
            payment_tier,
            age,
            gender,
            ever_benched,
            experience
        ]])

        prediction = model.predict(features)[0]

        confidence = None
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(features)[0]
            confidence = round(float(np.max(probs)) * 100, 2)

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Prediction error: {str(e)}"
        }), 400


# WSGI handler exposure for serverless platforms like Vercel
app_handler = app

if __name__ == '__main__':
    app.run(debug=True)
