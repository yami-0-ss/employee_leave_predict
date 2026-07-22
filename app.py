import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

# Get absolute project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')

# Initialize Flask with explicit template directory
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Safe model load
model = None
model_error = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        model_error = f"Model file missing at: {MODEL_PATH}"
except Exception as e:
    model_error = f"Failed to load pickle model: {str(e)}"


@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"<h3>Render Error: {str(e)}</h3>", 500


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'success': False,
            'error': model_error or 'Model instance not initialized.'
        }), 500

    try:
        # Check Form payload or JSON payload
        data = request.form if request.form else (request.get_json(silent=True) or {})

        # Parse inputs for the 8 features
        education = float(data.get('Education', 0))
        joining_year = float(data.get('JoiningYear', 2020))
        city = float(data.get('City', 0))
        payment_tier = float(data.get('PaymentTier', 1))
        age = float(data.get('Age', 25))
        gender = float(data.get('Gender', 0))
        ever_benched = float(data.get('EverBenched', 0))
        experience = float(data.get('ExperienceInCurrentDomain', 0))

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
            'error': f"Prediction computation failed: {str(e)}"
        }), 400


# Expose WSGI instance for Vercel auto-detection
app = app

if __name__ == '__main__':
    app.run(debug=True)
