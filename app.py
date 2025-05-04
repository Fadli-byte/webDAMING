from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)

# Load model (akan dibuat saat aplikasi pertama kali dijalankan)
def create_model():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Load dataset Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Simpan model
    with open('model/iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Simpan informasi fitur
    with open('model/iris_features.pkl', 'wb') as f:
        pickle.dump({
            'feature_names': iris.feature_names,
            'target_names': iris.target_names
        }, f)
    
    return model

# Pastikan folder model ada
if not os.path.exists('model'):
    os.makedirs('model')

# Load atau buat model
if os.path.exists('model/iris_model.pkl'):
    with open('model/iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/iris_features.pkl', 'rb') as f:
        features_info = pickle.load(f)
else:
    model = create_model()
    with open('model/iris_features.pkl', 'rb') as f:
        features_info = pickle.load(f)

@app.route('/')
def home():
    current_year = datetime.now().year
    return render_template('index.html', 
                          feature_names=features_info['feature_names'],
                          target_names=features_info['target_names'],
                          current_year=current_year)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        features = []
        for feature in features_info['feature_names']:
            feature_value = float(request.form.get(feature.replace(' ', '_'), 0))
            features.append(feature_value)
        
        # Prediksi
        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Format probabilitas untuk output
        probabilities = []
        for i, target in enumerate(features_info['target_names']):
            probabilities.append({
                'name': target,
                'probability': round(prediction_proba[i] * 100, 2)
            })
        
        result = {
            'prediction': features_info['target_names'][prediction],
            'probabilities': probabilities
        }
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/about')
def about():
    current_year = datetime.now().year
    return render_template('about.html', current_year=current_year)

if __name__ == '__main__':
    app.run(debug=True)