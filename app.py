import os
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
try:
    model_data = joblib.load('passenger_satisfaction_model.pkl')
    model = model_data['model']
    feature_names = model_data['feature_names']
    num_features = len(feature_names)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None
    feature_names = []
    num_features = 0

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', feature_names=feature_names, num_features=num_features)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    try:
        df = pd.read_csv(filepath)
        if df.shape[1] != num_features:
            return jsonify({'error': 'Invalid number of columns'}), 400
        preds = model.predict(df)
        results = [{'input': row.tolist(), 'prediction': str(pred)} for row, pred in zip(df.values, preds)]
        return jsonify({'feature_names': feature_names, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.get_json()
        if len(data['values']) != num_features:
            return jsonify({'error': 'Incorrect number of features'}), 400
        manual_input_df = pd.DataFrame([data['values']], columns=feature_names)
        prediction = model.predict(manual_input_df)[0]
        response = {'prediction': str(prediction)}
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(manual_input_df)[0]
            response['probabilities'] = {str(cls): float(p) for cls, p in zip(model.classes_, probabilities)}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)