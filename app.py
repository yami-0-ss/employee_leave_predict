from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained logistic regression model
model = pickle.load(open('logistic_model(1).pkl', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features in exact order expected by the model
        # Adjust categorical conversions based on how your model was trained (e.g., One-Hot or Label Encoding)
        education = int(request.form.get('Education', 0))
        joining_year = int(request.form.get('JoiningYear', 2022))
        city = int(request.form.get('City', 0))
        payment_tier = int(request.form.get('PaymentTier', 1))
        age = int(request.form.get('Age', 25))
        gender = int(request.form.get('Gender', 0))
        ever_benched = int(request.form.get('EverBenched', 0))
        exp_domain = int(request.form.get('ExperienceInCurrentDomain', 1))

        # Array of features
        features = np.array([[
            education, joining_year, city, payment_tier,
            age, gender, ever_benched, exp_domain
        ]])

        prediction = model.predict(features)[0]
        
        # Predict probability if supported
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            probability = round(float(np.max(proba)) * 100, 2)

        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'confidence': probability
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
