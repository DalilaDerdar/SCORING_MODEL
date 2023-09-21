#!/usr/bin/env python
# coding: utf-8

# In[2]:


import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline as imbPipeline
from imblearn.over_sampling import SMOTE
import streamlit as st
import joblib


# In[3]:


# Charger le modèle formé
@st.cache_data
def load_model():
    model = joblib.load('best_model.pkl')
    return model

# Prédire avec le modèle
def predict(model, data):
    y_pred_proba = model.predict_proba(data)[:, 1]
    threshold = 0.37
    y_pred = (y_pred_proba > threshold).astype(int)
    return y_pred

# Application principale
def main():
    st.title('Prêt à dépenser')
    st.markdown('## Outil de scoring crédit')
    st.markdown('### Prédire si un client est éligible à un prêt')
    st.markdown('#### Par: [Dalila Derdar](https://www.linkedin.com/in/daliladerdar)')

    # Charger le modèle
    model = load_model()

    # Charger les données d'échantillon pour la démo
    sample_df = pd.read_csv('../data/sample_df.csv')
    X_sample = sample_df.drop(columns=["TARGET"])
    
    # Créer une sidebar
    st.sidebar.title("Paramètres")
    index_selected = st.sidebar.selectbox("Choisissez le dossier client à tester en sélectionner son numéro de dossier:", X_sample.index)
    data_to_predict = X_sample.loc[[index_selected]]
    
    # Prédiction 
    pred = predict(model, data_to_predict)
    st.write(f'La prédiction pour l\'index {index_selected} est: {pred[0]}')

if __name__ == '__main__':
    main()

