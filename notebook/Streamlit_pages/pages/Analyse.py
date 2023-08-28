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
import shap
import plotly.graph_objects as go

# Fonction pour afficher les graphiques
def plot_parallel_bars(client_data, similar_clients_data, top_features):
    # Créer des données pour le graphique
    categories = top_features
    
    # Valeurs du client
    client_values = client_data[top_features].values.tolist()[0]
    
    # Valeurs moyennes des clients similaires
    similar_values = similar_clients_data[top_features].mean().tolist()

    # Créer le graphique
    fig = go.Figure(data=[
        go.Bar(name='Client', x=categories, y=client_values, marker_color='#CC00FF'),
        go.Bar(name='Clients Similaires (Moyenne)', x=categories, y=similar_values, marker_color='#0033FF')
    ])
    
    # Mise à jour de la disposition du graphique
    fig.update_layout(
        title="Comparaison du facteur clef, entre le client et les clients similaires",
        barmode='group',
        yaxis_title="Valeur",
        xaxis_title="Facteurs clefs",
        template="plotly_dark"  # Pour un thème moderne sombre
    )
    
    return fig

# Charger le modèle formé
@st.cache_data
def load_model():
    model = joblib.load('best_model.pkl')
    return model

def load_data():
    X = joblib.load('X.pkl')
    X_test = joblib.load('X_test.pkl')
    return X, X_test
X, X_test = load_data()

# Prédire avec le modèle
def predict(model, data):
    y_pred_proba = model.predict_proba(data)[:, 1]
    threshold = 0.37
    y_pred = (y_pred_proba > threshold).astype(int)
    return y_pred

# Application principale
def main():
    st.markdown("<h1 style='text-align: center; color: #800020 ;'>PRET A DEPENSER</h1>", unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    
    # Charger le modèle
    model = load_model()

    # Charger les données d'échantillon pour la démo
    sample_df = pd.read_csv('../../data/sample_df.csv')
    X_sample = sample_df.drop(columns=["TARGET"])
    
    # Créer une sidebar
    #st.sidebar.title("Paramètres")
    index_selected = st.sidebar.selectbox("""Choisissez le dossier client à tester en sélectionnant 
                                          son numéro de dossier:""", X_sample.index)
    data_to_predict = X_sample.loc[[index_selected]]
    #st.markdown(f'### RESULTAT D’ANALYSE DU DOSSIER CLIENT n°{index_selected}')
    st.markdown("""<h2 style='text-align: left; color: #5a5a5a;'>Résultat d'analyse du dossier client n°{}</h2>""".format(index_selected), 
                unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)


    # Créer une variable pour stocker l'état de la prédiction
    prediction_made = st.session_state.get("prediction_made", False)
    result = st.session_state.get("result", None)

    if st.sidebar.button('Lancer la prédiction'):
        pred = predict(model, data_to_predict)
        if pred[0] == 0:
            result = "accept"
        else:
            result = "refuse"
        prediction_made = True
        st.session_state.prediction_made = prediction_made
        st.session_state.result = result

    if prediction_made:
        if result == "accept":
            st.markdown("""
            <div style="background-color: green; padding: 10px 15px; border-radius: 5px; width: 50%; margin: 0 auto;">
                <h4 style="color: white; text-align: center;">Crédit accepté</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.write("""Votre client semble avoir les éléments requis pour rembourser son crédit durablement. 
                 Nous conseillons l’obtention du prêt.""")
            st.markdown("<br>", unsafe_allow_html=True)

        else:  # Si le résultat est "refuse"
            st.markdown("""
            <div style="background-color: red; padding: 10px 15px; border-radius: 5px; width: 50%; margin: 0 auto;">
                <h4 style="color: white; text-align: center;">Crédit refusé</h4>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<br>', unsafe_allow_html=True)
            st.write("""Votre client ne semble pas avoir les éléments nécessaires pour rembourser son crédit durablement. 
                 Nous ne conseillons pas l’obtention du prêt.""")
            st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.write("Veuillez cliquer sur le bouton pour obtenir la prédiction.")


    # Option "Information du client"
    if st.sidebar.checkbox("Informations du client"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Informations du client</h3>", unsafe_allow_html=True)

        with st.expander("Cliquez pour afficher les détails"):
            st.write("""Vous trouverez les informations du client qui ont permis de faire l'analyse de son dossier. 
                     Une vue globale pour permet de voir toutes les informations qui ont été prises en compte. 
                     Vous pouvez également sélectionner les colonnes qui vous intéressent pour un affichage détaillé.""")
        st.write(data_to_predict)
        st.markdown("<br>", unsafe_allow_html=True)

    # Sélection de colonnes spécifiques pour un affichage détaillé
        specific_columns = st.multiselect("Choisissez les colonnes pour un affichage détaillé:", sample_df.columns)
        if specific_columns:
            st.markdown("<h4>Détails sélectionnés</h4>", unsafe_allow_html=True)
            st.write(data_to_predict[specific_columns])
        st.markdown("<br>"*3, unsafe_allow_html=True)

    # Option "Facteurs clefs du client"
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.sidebar.checkbox("Facteurs clefs du client"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Facteurs clefs du client</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.markdown("""Comment lire ce graphique:<br>         
            <b>Axe Vertical:</b><br>
            Chaque point sur le graphique représente un facteur de votre 
            jeu de données, c'est-à-dire une information caractéristique de votre client.
            Les facteurs les plus influents sont en haut, et les moins influents sont en bas.
            <br>    
            <b>Axe Horizontal:</b><br>
            L'axe horizontal montre à quel point un facteur influence la prédiction.
            Si un point est à droite du centre (0), cela signifie que ce facteur augmente la 
            probabilité que le prêt ne soit pas remboursé et donc ne sera pas accordé. Au contraire, si le point est à gauche du centre, 
            cela signifie qu'il augmente la probabilié que le prêt soit remboursé et donc sera accordé.
            <br> 
                        <br> 
            <b>Couleur des Points:</b><br>
            La couleur des points donne une indication supplémentaire. Les points rouges indiquent des 
            valeurs élevées pour ce facteur et les points bleus des valeurs basses.
            <br>  
            <b>Densité des Points:</b><br>
            Là où vous voyez une concentration élevée de points (rouges ou bleus), cela signifie que cette 
            valeur particulière du facteur a un impact important pour beaucoup d'observations dans le jeu de données.
            <br>""",unsafe_allow_html=True)

        explainer = shap.TreeExplainer(model.named_steps['classifier'])
        shap_values = explainer.shap_values(data_to_predict)
        
        shap.initjs()
    
        # Générez votre plot
        shap.summary_plot(shap_values[1], data_to_predict, show=False)
        st.pyplot()
        st.markdown("<br>"*4, unsafe_allow_html=True)


    # Option "Information sur les facteurs clefs généraux"
    import plotly.express as px

    if st.sidebar.checkbox("Information sur les facteurs clefs généraux"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Information sur les facteurs clefs généraux</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.write(""" Cette section présente les facteurs clés qui ont le plus grand impact sur les décisions de prêt.
                    Chaque point du graphique représente une caractéristique de vos données. Plus le point est à droite,
                    plus cette caractéristique a un impact fort sur la prédiction de capacité de remboursement, et plus il est à gauche, plus elle a un impact faible.
                     Vous pouvez choisir le nombre de facteurs que vous souhaitez afficher.""")
        importances = model.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_sample.columns, 'Importance': importances})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=True)  # Inversez ici pour le tri ascendant

        # Ajouter une réglette pour choisir le nombre de caractéristiques à afficher
        num_features = st.slider('Nombre de facteurs clefs à afficher:', min_value=5, max_value=15, value=10, step=5)
        top_features = feature_importances[-num_features:]  # Prend les derniers éléments au lieu des premiers
        
        
    
        fig = px.bar(top_features, 
                 x='Importance', 
                 y='Feature', 
                 orientation='h',
                 labels={'Feature':'Features', 'Importance':'Importance'},
                 
                 title=f'Top {num_features} des facteurs clefs généraux',
                 color='Importance',
                 color_continuous_scale= 'bluered')  

        # Mettre à jour les titres des axes et les mettre en gras
        fig.update_layout(
            xaxis_title="<b>Niveau d'Importance</b>", 
            yaxis_title="<b>Facteurs clefs</b>"
)


        st.plotly_chart(fig)

    # Option "Comparaison des informations du client"
    if st.sidebar.checkbox("Comparer les informations du client"):
        st.markdown("<h3 style='text-align: left; color: #800020 ;'>Comparaison des informations du client</h3>", unsafe_allow_html=True)
        with st.expander("Cliquez pour afficher les détails"):
            st.write(""" Dans cette section, vous pouvez comparer les données de votre client avec les données de l'ensemble des clients ou avec 
                 les données des clients similaires. Pour chacune des comparaisons, les données des autres clients s'affichent sous forme de statistiques descriptives : 
                count pour compter le nombre de clients, mean pour la moyenne, std pour l'écart-type, min pour la valeur minimale, 25% pour le premier quartile, 
                50% pour la médiane, 75% pour le troisième quartile et max pour la valeur maximale.""")
        # Choix du groupe de comparaison
        compare_with = st.radio("Comparer avec :", ["Ensemble des clients", "Clients similaires"])
        
        # Obtenir les caractéristiques les plus importantes
        importances = model.named_steps['classifier'].feature_importances_
        feature_importances = pd.DataFrame({'Feature': X_sample.columns, 'Importance': importances})
        feature_importances_sorted = feature_importances.sort_values(by='Importance', ascending=False)
        top_features = feature_importances_sorted['Feature'][:10].tolist()  # Top 10 caractéristiques les plus importantes

        # Filtrer les données selon les caractéristiques les plus importantes
        X_filtered = X[top_features]
        data_to_predict_filtered = data_to_predict[top_features]

        # Afficher les statistiques descriptives
        if compare_with == "Ensemble des clients":
            st.write("Comparaison avec l'ensemble des clients:")
            desc = X_filtered.describe()
            st.write(desc)
            
            # Informations spécifiques du client par rapport à l'ensemble
            client_info = data_to_predict_filtered.describe()
            st.write(f"Informations pour le client n°{index_selected}:")
            st.write(data_to_predict_filtered)
            
        else:  # Clients similaires
            # Définir les clients similaires (10 plus proches)
            imputer = SimpleImputer(strategy='most_frequent')
            X_filled = pd.DataFrame(imputer.fit_transform(X_filtered), columns=X_filtered.columns)

            from sklearn.metrics.pairwise import euclidean_distances
            distances = euclidean_distances(X_filled, imputer.transform(data_to_predict_filtered))
            closest_indices = np.argsort(distances, axis=0)[:10]
            closest_clients = X_filled.iloc[closest_indices.flatten()]
            
            st.write("Comparaison avec les clients similaires:")
            desc_similar = closest_clients.describe()
            st.write(desc_similar)
            
            # Informations spécifiques du client par rapport aux clients similaires
            client_info = data_to_predict_filtered.describe()
            st.write(f"Informations pour le client n°{index_selected}:")
            st.write(data_to_predict_filtered)

            st.markdown("<br>"*3, unsafe_allow_html=True)

            #Graphique de comparaison
            # Choix de la caractéristique à comparer
            st.markdown("<h4 style='text-align: left; color: black ;'>Comparaison des facteurs clefs</h4>", unsafe_allow_html=True)
            with st.expander("Cliquez pour afficher les détails"):
                st.write(""" Choisissez et affichez sous forme de graphique, un facteur clef à comparer entre les données des clients similaires et celles du client sélectionné.""")
            feature_to_compare = st.selectbox("Choisissez le facteur clef :", top_features)
            fig = plot_parallel_bars(data_to_predict, closest_clients, [feature_to_compare])
            st.plotly_chart(fig)
    

if __name__ == '__main__':
    main()


