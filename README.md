# SAFEHOOD

## Contexte
La solution SAFEHOOD repose sur deux idées principales :
1. **Analyse automatique des mots sur les emballages d'aliments** : SAFEHOOD utilise des techniques de reconnaissance optique de caractères (OCR) pour extraire les informations textuelles des images des listes d'ingrédients figurant sur les emballages alimentaires. Cette fonctionnalité est particulièrement utile pour offrir aux consommateurs un accès facile et rapide aux détails des ingrédients des produits qu'ils souhaitent consommer, surtout lorsque ces informations sont présentées de manière condensée.

2. **Détection des allergènes** : SAFEHOOD rectifie les sorties générées par l'OCR et repère les différents allergènes présents dans la liste d'ingrédients. L'objectif est d'assurer la précision des informations fournies aux utilisateurs en mettant en évidence la présence d'aliments potentiellement dangereux pour certaines personnes. Pour ce faire, la solution intègre des techniques et des algorithmes issus du traitement du langage naturel (NLP - Natural Language Processing).

Cette fusion de la technologie OCR et du NLP améliore la compréhension des informations nutritionnelles, favorisant une alimentation plus éclairée et sécurisée.

## Structure du Répertoire GitHub
Le répertoire est structuré en quatre fichiers principaux, chacun représentant une étape de notre étude :

1. **0. Dataset_construction_nettoyage** :
   - **Description** : Ce fichier contient les codes utilisés pour explorer le dataset d'Open Food Facts, extraire les images et nettoyer ce dataset en éliminant les images trop petites et celles des codes-barres.
   - **Fonctions Clés** : Exploration et extraction des images, nettoyage des données.

2. **1. Prétraitement** :
   - **Description** : Ce fichier contient le code crucial pour notre étude. Il se concentre sur l'exploration et le test de différents prétraitements des images afin de construire une fonction qui retournera les meilleures sorties OCR.
   - **Méthodologie** : Utilisation de fonctions de traitement d'images telles que `adjust_brightness`, `improve_letter_readability`, `deskewing`, `apply_adaptive_binarization`, et `adjust_contrast_after_binarization`. Chaque étape de traitement est suivie par une évaluation visuelle de l'image et une extraction de texte via OCR pour juger de l'efficacité des prétraitements.
   - **Résultat** : Construction d'une base d'images et de textes de test à partir des images prétraitées. Utilisation de l'API Azure Cognitive Services pour construire une base de données des vérités terrain.

3. **2. NLP** :
   - **Description** : Ce fichier contient les codes initiaux pour appréhender le NLP, construire des dictionnaires d'allergènes depuis la base de données Open Food Facts, et traiter les données pour l'autocorrection et la détection d'allergènes.
   - **Fonctions Clés** : Séparation des données en dictionnaires français et anglais, normalisation des mots, nettoyage des données, utilisation de modèles NLP comme spaCy et NLTK pour détecter les allergènes dans des phrases spécifiques, autocorrection des erreurs.

4. **3. Final** :
   - **Description** : Ce fichier contient la plateforme de test en local, permettant de visualiser les résultats de l'océrisation en s'appuyant sur la méthode développée.
   - **Fonctions Clés** : Interface de test locale, visualisation des résultats OCR, validation des techniques développées.

## Installation et Utilisation
Clonez ce dépôt : 
    ```bash
    git clone https://github.com/votre_nom_de_utilisateur/safehood.git
    ```


## Contribution
Les contributions sont les bienvenues ! Pour contribuer :
1. Fork le dépôt.
2. Créez une branche pour vos modifications :
    ```bash
    git checkout -b ma-nouvelle-branche
    ```
3. Effectuez vos modifications et validez-les :
    ```bash
    git commit -m 'Ajout d'une nouvelle fonctionnalité'
    ```
4. Poussez vos modifications :
    ```bash
    git push origin ma-nouvelle-branche
    ```
5. Ouvrez une Pull Request.


Merci d'utiliser SAFEHOOD ! Pour toute question ou assistance, veuillez ouvrir une issue sur GitHub.
