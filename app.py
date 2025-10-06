from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from collections import Counter
import sys

# Install flask-ngrok if running in Colab
try:
    from flask_ngrok import run_with_ngrok
except ImportError:
    !pip install flask-ngrok -q
    from flask_ngrok import run_with_ngrok

app = Flask(__name__)

# -------------------- Load Models & Scaler --------------------
try:
    scaler = joblib.load('scaler.pkl')
    logreg = joblib.load('logistic_regression_model.pkl')
    svc_model = joblib.load('svm_model.pkl')
    dt_model = joblib.load('decision_tree_model.pkl')
    print("Models and Scaler loaded successfully.")
except FileNotFoundError:
    print("Error loading model or scaler files.")
    scaler, logreg, svc_model, dt_model = None, None, None, None

# -------------------- Feature Columns --------------------
feature_columns = [
    'Course', 'Daytime/evening attendance', 'Previous qualification',
    'Previous qualification (grade)', 'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)', 'GDP'
]

# -------------------- HTML Templates --------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Student Performance Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {background-color: #f0f8ff; font-family: 'Arial', sans-serif;}
        .container {margin-top: 40px; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
        h2 {color: #0d6efd; text-align: center; margin-bottom: 25px;}
        .form-label {font-weight: bold;}
        .btn-primary {background-color: #0d6efd; border-color: #0d6efd; padding: 10px 20px; font-size: 1.1rem;}
        .alert-info {color: #055160; background-color: #cff4fc; border-color: #b6effb;}
        small {color: #6c757d;}
    </style>
</head>
<body>
<div class="container">
    <h2>🎓 Student Performance Predictor</h2>
    {% if error %}
        <div class="alert alert-danger" role="alert">{{ error }}</div>
    {% endif %}
    <form action="/predict" method="post">
        <div class="row g-3">

            <div class="col-md-6">
                <label class="form-label">Course ID</label>
                <input type="number" class="form-control" name="Course" required placeholder="Enter a course number (e.g., 1-20)">
                <small>Choose a number corresponding to the course (1–20)</small>
            </div>

            <div class="col-md-6">
                <label class="form-label">Attendance Type</label>
                <select class="form-select" name="Daytime/evening attendance" required>
                    <option value="1">Daytime</option>
                    <option value="0">Evening</option>
                </select>
                <small>Select if the student attends Daytime or Evening classes</small>
            </div>

            <div class="col-md-6">
                <label class="form-label">Previous Qualification</label>
                <select class="form-select" name="Previous qualification" required>
                    <option value="1">High School</option>
                    <option value="2">Bachelor's</option>
                    <option value="3">Master's</option>
                    <option value="4">Other</option>
                </select>
            </div>

            <div class="col-md-6">
                <label class="form-label">Previous Qualification Grade</label>
                <input type="number" class="form-control" name="Previous qualification (grade)" step="0.01" min="0" max="20" required placeholder="Enter grade 0–20">
                <small>Enter grade from 0 to 20</small>
            </div>

            <div class="col-md-6">
                <label class="form-label">1st Semester: Credited Units</label>
                <input type="number" class="form-control" name="Curricular units 1st sem (credited)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">1st Semester: Evaluations</label>
                <input type="number" class="form-control" name="Curricular units 1st sem (evaluations)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">1st Semester: Approved Units</label>
                <input type="number" class="form-control" name="Curricular units 1st sem (approved)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">1st Semester: Grade</label>
                <input type="number" class="form-control" name="Curricular units 1st sem (grade)" step="0.01" min="0" max="20" required placeholder="0–20">
            </div>

            <div class="col-md-6">
                <label class="form-label">1st Semester: Units without Evaluations</label>
                <input type="number" class="form-control" name="Curricular units 1st sem (without evaluations)" step="0.01" min="0" max="20" required placeholder="0–20">
            </div>

            <div class="col-md-6">
                <label class="form-label">2nd Semester: Credited Units</label>
                <input type="number" class="form-control" name="Curricular units 2nd sem (credited)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">2nd Semester: Evaluations</label>
                <input type="number" class="form-control" name="Curricular units 2nd sem (evaluations)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">2nd Semester: Approved Units</label>
                <input type="number" class="form-control" name="Curricular units 2nd sem (approved)" min="0" max="60" required placeholder="0–60">
            </div>

            <div class="col-md-6">
                <label class="form-label">2nd Semester: Grade</label>
                <input type="number" class="form-control" name="Curricular units 2nd sem (grade)" step="0.01" min="0" max="20" required placeholder="0–20">
            </div>

            <div class="col-md-6">
                <label class="form-label">GDP (Approx.)</label>
                <input type="number" class="form-control" name="GDP" step="0.01" min="0" required placeholder="Enter GDP in USD">
                <small>Enter approximate GDP value (e.g., 20000)</small>
            </div>

        </div>

        <div class="text-center mt-4">
            <button type="submit" class="btn btn-primary">Predict Performance</button>
        </div>
    </form>
</div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {background-color: #f0f8ff; font-family: 'Arial', sans-serif;}
        .container {margin-top: 50px; background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
        h3 {color: #28a745; text-align: center; margin-bottom: 30px;}
        .alert-info {color: #055160; background-color: #cff4fc; border-color: #b6effb; font-size: 1.3rem; font-weight: bold; text-align:center;}
        .btn-outline-primary {border-color: #0d6efd; color: #0d6efd;}
        .btn-outline-primary:hover {background-color: #0d6efd; color: white;}
    </style>
</head>
<body>
<div class="container">
    <h3>📊 Student Performance Prediction</h3>

    <div class="alert alert-info">
        🎯 Predicted Result: <strong>{{ final_label }}</strong>
    </div>

    <div class="text-center mt-3">
        <a href="/" class="btn btn-outline-primary">🔁 Predict Another Student</a>
    </div>
</div>
</body>
</html>
"""

# -------------------- Routes --------------------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML, request=request)

@app.route('/predict', methods=['POST'])
def predict():
    if scaler is None or logreg is None or svc_model is None or dt_model is None:
        error_message = "Models or scaler not loaded. Please check the notebook output."
        return render_template_string(INDEX_HTML, error=error_message, request=request), 500

    try:
        user_input = {}
        for col in feature_columns:
            if col == 'Daytime/evening attendance' or col == 'Previous qualification':
                user_input[col] = int(request.form[col])
            elif col in ['Previous qualification (grade)', 'Curricular units 1st sem (grade)',
                         'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (grade)', 'GDP']:
                user_input[col] = float(request.form[col])
            else:
                user_input[col] = int(request.form[col])

        new_data = pd.DataFrame([user_input])[feature_columns]
        new_data_scaled = scaler.transform(new_data)

        # Predictions (hide individual models, only final)
        pred_logreg = logreg.predict(new_data_scaled)[0]
        pred_svc = svc_model.predict(new_data_scaled)[0]
        pred_dt = dt_model.predict(new_data_scaled)[0]

        predictions_raw = [pred_logreg, pred_svc, pred_dt]
        prediction_counts = Counter(predictions_raw)
        majority_pred = prediction_counts.most_common(1)[0][0]

        labels = {0: 'Fail/Dropout', 1: 'Pass', 2: 'None'}
        final_label = labels.get(majority_pred, 'Unknown')

        return render_template_string(RESULT_HTML, final_label=final_label)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}", exc_info=True)
        error_message = f"An error occurred during prediction: {e}"
        return render_template_string(INDEX_HTML, error=error_message, request=request), 500

# -------------------- Run App --------------------
if 'google.colab' in sys.modules:
    print("Running in Google Colab, using ngrok.")
    run_with_ngrok(app)
else:
    app.run(debug=True)