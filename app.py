#Run on environnement Python : 3.10.11.64bit\AppDAta\Local\Program\Python

from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
import shap
import os

app = Flask(__name__)
PORT = int(os.environ.get('PORT', 5000))

# Charger le modèle formé
def load_model():
    model = joblib.load('best_model.pkl')
    return model
model = load_model()

explainer = joblib.load('explainer.pkl')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = json.loads(request.get_json())
        selected_row = pd.DataFrame(data, index=[0])

        prediction_proba = model.predict_proba(selected_row)
        threshold = 0.37
        prediction = int(prediction_proba[0][1] > threshold)

        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
    
# Route pour l'explication
@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = json.loads(request.get_json())
        selected_row = pd.DataFrame(data, index=[0])

        shap_values = explainer.shap_values(selected_row)

        # Prendre seulement les valeurs SHAP pour la classe 1 (ou 0 si vous voulez la classe 0)
        shap_values_class1 = shap_values[1]
        
        # Convertir les éléments de shap_values_class1 en listes
        shap_values_as_list = shap_values_class1.tolist()

        # Convertir les éléments de shap_values en listes
        # shap_values_as_lists = [elem.tolist() for elem in shap_values]
        
        return jsonify(shap_values_as_list)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=PORT)

