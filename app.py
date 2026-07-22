import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# 1. Absolute Base Directory Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'logistic_model.pkl')

app = Flask(__name__)

# 2. Safe Model Loading
model = None
model_error = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        model_error = f"Error unpickling model: {str(e)}"
else:
    model_error = f"Model file not found at path: {MODEL_PATH}"


# 3. HTML UI Embedded Directly to Prevent Vercel File Path Failures
HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Analytics & Prediction</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-purple: #8b5cf6;
            --accent-pink: #ec4899;
            --accent-cyan: #06b6d4;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
        body { background: var(--bg-gradient); min-height: 100vh; color: var(--text-main); display: flex; align-items: center; justify-content: center; padding: 2rem 1rem; }
        .container { width: 100%; max-width: 900px; background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--card-border); border-radius: 24px; padding: 2.5rem; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
        .header { text-align: center; margin-bottom: 2.5rem; }
        .header h1 { font-size: 2.25rem; font-weight: 700; background: linear-gradient(to right, #a78bfa, #f472b6, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
        .header p { color: var(--text-muted); font-size: 0.95rem; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.25rem; }
        .input-group { display: flex; flex-direction: column; gap: 0.5rem; }
        .input-group label { font-size: 0.85rem; font-weight: 600; color: #cbd5e1; }
        .input-group input, .input-group select { width: 100%; padding: 0.75rem 1rem; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--card-border); border-radius: 12px; color: #fff; font-size: 0.95rem; outline: none; }
        .input-group input:focus, .input-group select:focus { border-color: var(--accent-purple); box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25); }
        .btn-submit { grid-column: 1 / -1; margin-top: 1rem; padding: 1rem; border: none; border-radius: 12px; background: linear-gradient(135deg, var(--accent-purple), var(--accent-pink)); color: white; font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s ease; }
        .btn-submit:hover { transform: translateY(-2px); }
        .result-box { margin-top: 2rem; padding: 1.5rem; border-radius: 16px; background: rgba(15, 23, 42, 0.8); border: 1px solid var(--card-border); display: none; text-align: center; }
        .result-box.active { display: block; }
        .result-value { font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; color: var(--accent-cyan); }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Logistic Predictor Portal</h1>
        <p>Enter parameters to compute real-time predictions</p>
    </div>

    <form id="predictionForm">
        <div class="form-grid">
            <div class="input-group">
                <label for="Education">Education Level</label>
                <select name="Education" id="Education" required>
                    <option value="0">Bachelors (0)</option>
                    <option value="1">Masters (1)</option>
                    <option value="2">PHD (2)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="JoiningYear">Joining Year</label>
                <input type="number" name="JoiningYear" id="JoiningYear" value="2017" required>
            </div>

            <div class="input-group">
                <label for="City">City Code</label>
                <select name="City" id="City" required>
                    <option value="0">Bangalore (0)</option>
                    <option value="1">Pune (1)</option>
                    <option value="2">New Delhi (2)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="PaymentTier">Payment Tier (1-3)</label>
                <input type="number" name="PaymentTier" id="PaymentTier" min="1" max="3" value="3" required>
            </div>

            <div class="input-group">
                <label for="Age">Age</label>
                <input type="number" name="Age" id="Age" value="28" required>
            </div>

            <div class="input-group">
                <label for="Gender">Gender</label>
                <select name="Gender" id="Gender" required>
                    <option value="0">Male (0)</option>
                    <option value="1">Female (1)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="EverBenched">Ever Benched?</label>
                <select name="EverBenched" id="EverBenched" required>
                    <option value="0">No (0)</option>
                    <option value="1">Yes (1)</option>
                </select>
            </div>

            <div class="input-group">
                <label for="ExperienceInCurrentDomain">Domain Experience (Years)</label>
                <input type="number" name="ExperienceInCurrentDomain" id="ExperienceInCurrentDomain" value="3" required>
            </div>

            <button type="submit" class="btn-submit">Run Prediction Model</button>
        </div>
    </form>

    <div id="resultBox" class="result-box">
        <span style="color: var(--text-muted); font-size: 0.9rem;">Prediction Output</span>
        <div id="resultValue" class="result-value">Class 0</div>
        <div id="confidenceValue" style="color: var(--text-muted); margin-top: 0.25rem; font-size: 0.85rem;"></div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(e.target);
        const resultBox = document.getElementById('resultBox');
        const resultValue = document.getElementById('resultValue');
        const confidenceValue = document.getElementById('confidenceValue');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                resultBox.classList.add('active');
                resultValue.textContent = `Predicted Class: ${data.prediction}`;
                if (data.confidence) {
                    confidenceValue.textContent = `Model Confidence: ${data.confidence}%`;
                }
            } else {
                alert('Error: ' + data.error);
            }
        } catch (err) {
            alert('Server Error!');
        }
    });
</script>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_UI)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': model_error or "Model not loaded"}), 500

    try:
        data = request.form if request.form else (request.get_json(silent=True) or {})

        education = float(data.get('Education', 0))
        joining_year = float(data.get('JoiningYear', 2020))
        city = float(data.get('City', 0))
        payment_tier = float(data.get('PaymentTier', 1))
        age = float(data.get('Age', 25))
        gender = float(data.get('Gender', 0))
        ever_benched = float(data.get('EverBenched', 0))
        experience = float(data.get('ExperienceInCurrentDomain', 0))

        features = np.array([[
            education, joining_year, city, payment_tier,
            age, gender, ever_benched, experience
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
        return jsonify({'success': False, 'error': str(e)}), 400

# WSGI Handler export
app = app

if __name__ == '__main__':
    app.run(debug=True)
