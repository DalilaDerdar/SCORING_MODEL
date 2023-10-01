Scoring_Model : l'outil de scoring bancaire

L'outil scoring bancaire est un outil de prédiction de faillite de clients demandant un crédit. 
L'outil se matérialise sous forme de web app disponible ici  https://scoringmodel-m678rmdx2w6evqd4sdrfmv.streamlit.app/

Vous trouverez dans ce repositories :

--> Un dossier notebook comportant : 

    - les notebooks de modelisation 
        ¤ EDA : EDA_all_df.ipynb
        ¤ prétraitement : create_final_df.ipynb
        ¤ modélisation intégrant via MLFlow le tracking et stockage des modèles: modelisation.ipynb 

    - un sous dossier Streamlit où trouver le code générant le dashboard 
        ¤ Home.py
        ¤ sous dossier pages : 1_Analyse.py et 2_Glossaire.py

    - un fichier html testant un éventuel data drift sur nos données
        ¤ data_drift.html

        
--> Un script python pour l'API:

      - app.py

--> un dossier .github/workflows comportant:

    - un fichier YAML de déploiement de l'API sur Azure webapp

--> un script texte pour tous les packages utilisés dans ce projet

    - requirements.txt
