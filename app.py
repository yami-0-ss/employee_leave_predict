from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained logistic regression model
model = joblib.load('logistic_model.pkl')

# Feature Encoders / Mappers
# Adjust these dictionary mappings if your model used different numerical encodings during training
EDUCATION_MAP = {'Bachelors': 0, 'Masters': 1, 'PHD': 2}
CITY_MAP = {'Bangalore': 0, 'Pune': 1, 'New Delhi': 2}
GENDER_MAP = {'Male': 0, 'Female': 1}
EVER_BENCHED_MAP = {'No': 0, 'Yes': 1}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    prediction_class = None

    if request.method == 'POST':
        try:
            # Extract inputs from HTML form
            education = EDUCATION_MAP.get(request.form['Education'], 0)
            joining_year = int(request.form['JoiningYear'])
            city = CITY_MAP.get(request.form['City'], 0)
            payment_tier = int(request.form['PaymentTier'])
            age = int(request.form['Age'])
            gender = GENDER_MAP.get(request.form['Gender'], 0)
            ever_benched = EVER_BENCHED_MAP.get(request.form['EverBenched'], 0)
            experience = int(request.form['ExperienceInCurrentDomain'])

            # Combine features into numpy array in exact model order:
            # ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain']
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

            # Predict
            prediction = model.predict(features)[0]

            # Custom output message based on binary prediction
            if prediction == 1:
                prediction_text = "Prediction Result: Positive / Class 1"
                prediction_class = "success"
            else:
                prediction_text = "Prediction Result: Negative / Class 0"
                prediction_class = "warning"

        except Exception as e:
            prediction_text = f"Error in processing inputs: {str(e)}"
            prediction_class = "error"

    return render_template('index.html', prediction_text=prediction_text, prediction_class=prediction_class)

if __name__ == '__main__':
    app.run(debug=True)
