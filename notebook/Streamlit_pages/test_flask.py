#difficulté dans le test de la fonction predict de l'api flask : format des données envoyées 
#à l'api. Selected_row doit être un dictionnaire, et non un dataframe avant d'être transformé 
#en json et envoyé à l'api.

import unittest
from flask import json
import sys
sys.path.insert(0, 'notebook/Streamlit_pages')  # Il faut pointer vers le dossier, pas le fichier directement
from flask_app import app 
import pandas as pd
import requests


class TestFlask(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.app = app.test_client()
        cls.sample_df = pd.read_csv('../../data/sample_df.csv')  
        cls.X_sample = cls.sample_df.drop(columns=["TARGET"])
        print(type(cls.X_sample))

    def setUp(self):
        self.success_samples = self.X_sample.sample(n=10)
        print(type(self.success_samples))

    def test_predict_success(self):
        # Sélectionner une seule ligne du dataframe
        selected_row = self.success_samples.iloc[0]
        print(selected_row.isna().sum())
        json_data = json.dumps(selected_row.to_dict())#transforme la ligne en dictionnaire avant d'être en json 

        # URL de votre API Flask
        url = "http://localhost:5000/predict"
        
        # Envoyer la requête POST à l'API Flask
        response = requests.post(url, json=json_data)
        print(response.status_code)
        print(response.text)
        
        # Convertir la réponse en JSON
        response_data = response.json()

        # Vérifier que la réponse est soit 0, soit 1
        self.assertIn(response_data, [0, 1]) #nous cherchons juste à tester si on a bien 0 ou 1 en réponse

    
if __name__ == '__main__':
    unittest.main()
