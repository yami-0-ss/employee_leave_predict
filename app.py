import os
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Resolve absolute path for Vercel serverless environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')

# Load model safely
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

# Feature Encoders / Mappers matching dataset conventions[cite: 1]
EDUCATION_MAP = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}[cite: 1]
CITY_MAP = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}[cite: 1]
GENDER_MAP = {'Male': 0, 'Female': 1}[cite: 1]
EVER_BENCHED_MAP = {'No': 0, 'Yes': 1}[cite: 1]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_class = None

    if request.method == 'POST':
        if model is None:
            return render_template(
                'index.html',
                prediction_text="Error: Could not load 'logistic_model.pkl' on the server.",
                prediction_class="error"
            )

        try:
            # Extract inputs safely
            education = EDUCATION_MAP.get(request.form.get('Education'), 0)
            joining_year = int(request.form.get('JoiningYear', 2020))
            city = CITY_MAP.get(request.form.get('City'), 0)
            payment_tier = int(request.form.get('PaymentTier', 1))
            age = int(request.form.get('Age', 25))
            gender = GENDER_MAP.get(request.form.get('Gender'), 0)
            ever_benched = EVER_BENCHED_MAP.get(request.form.get('EverBenched'), 0)
            experience = int(request.form.get('ExperienceInCurrentDomain', 0))

            # Feature array: ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
            features = np.array([[
                education,
                joining_year,
                city,
                payment_tier,
                age,
                gender,
                ever_benched,
                experience
            ]], dtype=np.float64)

            # Predict
            prediction = model.predict(features)[0]

            if int(prediction) == 1:
                prediction_text = "Prediction Result: Employee likely to leave / High Risk (Class 1)"
                prediction_class = "warning"
            else:
                prediction_text = "Prediction Result: Employee likely to stay / Low Risk (Class 0)"
                prediction_class = "success"

        except Exception as e:
            prediction_text = f"Prediction Error: {str(e)}"
            prediction_class = "error"

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        prediction_class=prediction_class
    )

# Required for local testing
if __name__ == '__main__':
    app.run(debug=True)
