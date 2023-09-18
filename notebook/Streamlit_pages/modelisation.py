# %% [markdown]
# ## Pipeline to search best model

# %%
# import mlflow
# from mlflow.tracking import MlflowClient
# import mlflow.sklearn
import numpy as np
import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler
# from time import time
# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
# from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, learning_curve, validation_curve
# from sklearn.neighbors import KNeighborsClassifier
# from scipy.stats import randint
# import matplotlib.pyplot as plt
# from imblearn.pipeline import Pipeline as imbPipeline
# from imblearn.over_sampling import SMOTE

import os
os.chdir('C:/Users/DalilaDerdar/OneDrive - Supplier Assessment Services Limited/Bureau/Scoring_Model')


# %% [markdown]
# ### Pipeline 3-8 - Randomforest, test hyperparamètres - pipeline preprocessing & classifier + ML FLow + SMOTE - métric métier

# %%
# from imblearn.pipeline import Pipeline as imbPipeline
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import make_scorer, confusion_matrix

# Custom metric
def custom_cost_only_fn(y_true, y_pred):
    """
    Return the custom cost function for the test set.   
    Range: [-1, 1], -1 
    Args:
        y_true (array-like): True labels.   
        y_pred (array-like): Predicted labels.  
    Returns:
        float: Custom cost function.
    """
    y_true = np.array(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (1*tn - 10*fn) / y_true.shape[0]


# # %%

# custom_scorer_fn_only = make_scorer(custom_cost_only_fn, greater_is_better=True)

# # Read data
# sample_df = pd.read_csv('data/sample_df.csv')

# # Replace infinite values with NaN
# sample_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# # Separate the target from the rest of the data
# X = sample_df.drop(columns=["TARGET"])
# y = sample_df['TARGET']

# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Setup MLFLOW
# mlflow.set_experiment('Scoring_models')
# mlflow.sklearn.autolog()

# # Initialize SMOTE
# smote = SMOTE(random_state=42)

# # Define classifier
# clf = RandomForestClassifier(random_state=42)

# # Combine preprocessing, SMOTE, and classifier into a single pipeline
# pipeline = imbPipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('scaler', RobustScaler()),
#     ('smote', smote),
#     ('classifier', clf)
# ])

# # Define hyperparameters for the Random Forest Classifier
# param_dist = {
#     'classifier__n_estimators': randint(100, 200),
#     'classifier__max_depth': randint(1, 5),
#     'classifier__class_weight': ['balanced', 'balanced_subsample'],
#     'classifier__min_samples_split': randint(20, 40),
#     'classifier__min_samples_leaf': randint(10, 40),
#     'classifier__max_features': ['sqrt', 'log2', None]
# }

# # Add your custom scorer to the list of scorings
# scorings = {
#     'roc_auc': 'roc_auc',
#     'precision': 'precision',
#     'recall': 'recall',
#     'f1': 'f1',
#     'custom_score': custom_scorer_fn_only
# }

# # Initialize RandomizedSearchCV with your custom scorer
# random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, cv=5, n_jobs=-1, scoring=scorings, refit='roc_auc', n_iter=35, random_state=42)

# # Start an MLflow run
# with mlflow.start_run(run_name='Modelisation_custom_metric_10'):

#     # Train the model
#     random_search.fit(X_train, y_train)

#     # Get the best model
#     best_model = random_search.best_estimator_

#     # Serialisation du meilleur modèle
#     import joblib
#     joblib.dump(best_model, 'best_model.pkl')
#     joblib.dump(X, 'X.pkl')
#     joblib.dump(X_test, 'X_test.pkl')


#     # Calculate train score
#     train_score = roc_auc_score(y_train, best_model.predict(X_train))

#     # Calculate validation score
#     validation_score = random_search.best_score_

#     # Predict the labels of the test set
#     y_pred = random_search.predict(X_test)

#     # Evaluate on the test set
#     test_score = roc_auc_score(y_test, y_pred)

#     # Calculate custom metric
#     test_custom_cost = custom_cost_only_fn(y_test, y_pred)

#     # Calculate custom metric for the training set
#     train_custom_cost = custom_cost_only_fn(y_train, best_model.predict(X_train))

#     # Predict the probabilities of the test set
#     y_pred_proba = best_model.predict_proba(X_test)[:, 1]
#     threshold = 0.37 #par défaut, fixé à 0.5
#     y_pred = (y_pred_proba > threshold).astype(int)

#     # Calculate other metrics
#     auc = roc_auc_score(y_test, y_pred_proba)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)

#     # Log the parameters
#     mlflow.log_params(random_search.best_params_)

#     # Log the metrics
#     mlflow.log_metric("Training ROC AUC", train_score)
#     mlflow.log_metric("Validation ROC AUC", validation_score)
#     mlflow.log_metric("Test ROC AUC", test_score)
#     mlflow.log_metric("Test AUC", auc)
#     mlflow.log_metric("Precision", precision)
#     mlflow.log_metric("Recall", recall) 
#     mlflow.log_metric("F1 Score", f1)
#     mlflow.log_metric("Training Custom Metric", train_custom_cost)
#     mlflow.log_metric("Test Custom Metric", test_custom_cost)

#     # Log the model
#     mlflow.sklearn.log_model(best_model, "model")

#     # Store the results in a DataFrame
#     results = pd.DataFrame({
#         'Model': [random_search.best_estimator_.named_steps['classifier'].__class__.__name__],
#         'Best Parameters': [random_search.best_params_],
#         'Training ROC AUC': [train_score],
#         'Validation ROC AUC': [validation_score],
#         'Test ROC AUC': [test_score],
#         'Test AUC': [auc],
#         'Precision': [precision],
#         'Recall': [recall],
#         'F1 Score': [f1],
#         'Training Custom Metric': [train_custom_cost],
#         'Test Custom Metric': [test_custom_cost]
#     })

# print(results)


# # %%
# cm_smote_metier = confusion_matrix(y_test, y_pred)
# print(cm_smote_metier)

# # %% [markdown]
# # #### Comparaison des résultats du custom metric

# # %% [markdown]
# # ##### Custom metric pour un modèle parfait 

# # %% [markdown]
# # Profit_bycustomer = (1*tn - 10*fn) / y_true.shape[0]
# # 
# # Sur  base de cette CM 
# # 
# # [[522 271]
# # 
# #  [ 27  43]]
# # 
# # 
# # CM Parfaite (90% qui rembourse bien leur crédit - (tn) et 10% qui ne rembourse pas leur crédit (tp)
# # 
# # [[777 (TN), 0 (FN)],
# # 
# # [0 (FP), 86 (TP)]]
# # 
# # Calcul de la métrique métier
# # 
# # Profit_bycustomer = (1*793 - 10*0) / 863
# # 
# # Profit_bycustomer = 0.90

# # %% [markdown]
# # ##### Custom metric pour un modèle naif

# # %% [markdown]
# # CM Naive (90% qui rembourse bien leur crédit - (TN) et 10% de faux négatif (FN)).
# # 
# # [[777 (TN), 86 (FN)],
# # [0 (FP), 0 (TP)]]
# # 
# # Calcul de la métrique métier
# # 
# # Profit_bycustomer = (1*777 - 10*86) / 863
# # Profit_bycustomer = -0,096
# # 

# # %% [markdown]
# # ### Pipeline 3-9 - DummyClassifier, ML FLow + SMOTE - métric métier

# # %%
# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import roc_auc_score
# import mlflow

# # Initialize Dummy Classifier
# dummy_clf = DummyClassifier(strategy="stratified", random_state=42)

# # Pipeline for Dummy Classifier
# dummy_pipeline = imbPipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('scaler', RobustScaler()),
#     ('smote', smote),
#     ('classifier', dummy_clf)
# ])

# # Start an MLflow run
# with mlflow.start_run(run_name='Dummy_model'):

#     # Train the model
#     dummy_pipeline.fit(X_train, y_train)

#     # Predict the labels of the train set
#     y_train_pred_dummy = dummy_pipeline.predict(X_train)

#     # Calculate train score
#     train_score_dummy = roc_auc_score(y_train, y_train_pred_dummy)

#     # Predict the labels of the test set
#     y_test_pred_dummy = dummy_pipeline.predict(X_test)

#     # Evaluate on the test set
#     test_score_dummy = roc_auc_score(y_test, y_test_pred_dummy)

#     # Log the metrics
#     mlflow.log_metric("Training ROC AUC", train_score_dummy)
#     mlflow.log_metric("Test ROC AUC", test_score_dummy)

#     # Log the model
#     mlflow.sklearn.log_model(dummy_clf, "dummy_model")

#     # Store the results in a DataFrame
#     dummy_results = pd.DataFrame({
#         'Model': ['DummyClassifier'],
#         'Training ROC AUC': [train_score_dummy],
#         'Test ROC AUC': [test_score_dummy]
#     })

# print(dummy_results)


# # %% [markdown]
# # ## Importances des features globales

# # %%
# # Obtenir les importances des caractéristiques du modèle
# importances = best_model.named_steps['classifier'].feature_importances_

# # Créer un DataFrame pour visualiser les importances des caractéristiques
# feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# # Trier le DataFrame en fonction des importances
# feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# # Afficher les importances
# print(feature_importances)


# # %%
# # Limiter le DataFrame aux 100 caractéristiques les plus importantes
# top_30_features = feature_importances[:30]

# # Créer un diagramme à barres pour les importances des caractéristiques
# plt.figure(figsize=(10,8))
# plt.barh(top_30_features['Feature'], top_30_features['Importance'], color='b', align='center')
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.title('Top 50 Feature Importances')
# plt.gca().invert_yaxis()  # Inverser l'axe y pour afficher la caractéristique la plus importante en haut
# plt.show()


# # %% [markdown]
# # ## Importance des features locales 

# # %%
# # Importer la bibliothèque SHAP
# import shap

# # Initialiser le "Tree Explainer" de SHAP
# explainer = shap.TreeExplainer(best_model.named_steps['classifier'])

# # Calculer les valeurs SHAP pour les features de votre ensemble de test
# shap_values = explainer.shap_values(X_test)

# import joblib
# joblib.dump(explainer, 'explainer.pkl')

# # Visualiser les valeurs SHAP pour la première prédiction
# shap.initjs()

# # Afficher le graphique
# shap.force_plot(explainer.expected_value[1], shap_values[1][0,:], X_test.iloc[0,:])

# # %%
# # Créer un graphique d'importance des features pour l'ensemble du test
# shap.summary_plot(shap_values[1], X_test)

# # %% [markdown]
# # ## Test Data drift

# # %%
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset

# reference_data = pd.read_csv("../data/application_train.csv").drop(columns = ["TARGET"]).sample(10000)
# current_data = pd.read_csv("../data/application_test.csv").sample(10000)

# data_drift_report = Report(metrics=[DataDriftPreset()])

# data_drift_report.run(current_data=current_data, reference_data=reference_data, column_mapping=None)
# data_drift_report.save_html("data_drift.html")


