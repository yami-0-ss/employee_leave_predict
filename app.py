import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the pickle model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model file not found!'}), 500

    try:
        # Extract features in the required order
        education = float(request.form.get('Education', 0))
        joining_year = float(request.form.get('JoiningYear', 2020))
        city = float(request.form.get('City', 0))
        payment_tier = float(request.form.get('PaymentTier', 1))
        age = float(request.form.get('Age', 25))
        gender = float(request.form.get('Gender', 0))
        ever_benched = float(request.form.get('EverBenched', 0))
        experience = float(request.form.get('ExperienceInCurrentDomain', 0))

        # Format features into 2D array for scikit-learn model
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
        
        # Calculate probability if model supports predict_proba
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = round(float(np.max(probabilities)) * 100, 2)

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
