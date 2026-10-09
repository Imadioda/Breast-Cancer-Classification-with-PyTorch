# Breast Cancer Classification with PyTorch

## Présentation du projet

Ce projet consiste à développer un modèle de classification binaire permettant de prédire si une tumeur du sein est **bénigne (B)** ou **maligne (M)** à partir de caractéristiques numériques.

Réalisé avec Python et PyTorch, ce projet couvre les principales étapes d'un pipeline de machine learning : prétraitement des données, construction d'un réseau de neurones, entraînement et évaluation des performances.

## Jeu de données

Le projet utilise le fichier `BreastCancer.csv`, qui contient **569 individus et 33 colonnes** avant le prétraitement.

Il s'agit d'un petit jeu de données adapté à l'expérimentation de modèles de classification supervisée.

Le prétraitement comprend :

* La suppression de la colonne `id` et des colonnes inutiles.
* L'encodage de la variable cible : `Bénin = 0`, `Malin = 1`.
* La standardisation des variables explicatives avec `StandardScaler`.
* Le filtrage des variables fortement corrélées, avec un seuil de corrélation absolue supérieur à 0,9.

Après prétraitement, **20 variables explicatives** sont utilisées comme entrées du modèle.

## Architecture du réseau de neurones

Le modèle est un réseau de neurones entièrement connecté composé de deux couches linéaires.

```text
Entrée : 20 variables
        |
   Linear(20, 10)
        |
       ReLU
        |
  BatchNorm1d(10)
        |
   Dropout(0.3)
        |
   Linear(10, 1)
        |
      Logit
```

### Paramètres du modèle

* **Couche d'entrée :** 20 variables
* **Couche cachée :** 10 neurones
* **Couche de sortie :** 1 neurone
* **Fonction d'activation :** ReLU
* **Normalisation :** Batch Normalization
* **Régularisation :** Dropout de 30 %
* **Fonction de perte :** `BCEWithLogitsLoss`
* **Seuil de classification :** 0,5

La fonction `BCEWithLogitsLoss` est appliquée directement à la sortie brute du réseau (*logit*). Une fonction sigmoïde est utilisée lors de l'évaluation pour convertir cette sortie en probabilité.

## Entraînement

Les données sont réparties aléatoirement en trois ensembles :

* **70 %** pour l'entraînement ;
* **15 %** pour la validation ;
* **15 %** pour le test.

L'entraînement utilise des mini-batchs de 32 individus. Un mécanisme d'**early stopping** permet d'interrompre l'entraînement lorsque la perte de validation ne s'améliore plus pendant plusieurs époques.

L'ensemble de test est utilisé pour évaluer les performances finales du modèle.

## Évaluation et analyse des performances

Le modèle est évalué à l'aide de plusieurs métriques :

* Accuracy
* Sensibilité (*Recall*)
* Spécificité
* Vrais positifs (VP)
* Vrais négatifs (VN)
* Faux positifs (FP)
* Faux négatifs (FN)
* Courbe ROC
* Aire sous la courbe ROC (AUC)

Des visualisations permettent également d'analyser les probabilités prédites et les erreurs de classification.

## Résultats

Résultats obtenus lors d'un lancement du modèle :

| Métrique                        | Résultat |
| ------------------------------- | -------: |
| Loss d'entraînement (moyenne)   |   0.1567 |
| Loss de validation  (moyenne)   |   0.1859 |
| Accuracy de validation          |  96.84 % |
| Accuracy de test                |  95.12 % |

Les valeurs de sensibilité, de spécificité et d'AUC sont calculées par le script `ModelAnalysis.py`.

> **Remarque :** les résultats peuvent varier d'un lancement à l'autre en raison des opérations aléatoires utilisées lors de l'entraînement, du mélange des données et du Dropout. Les résultats présentés correspondent à une exécution particulière et ne garantissent pas les mêmes performances à chaque lancement.

## Structure du projet

```text
BreastCancer-ML/
├── code.py
│   ├── DatasetClean
│   ├── NeuralNetworkBC
│   └── ModelAnalysis
├── BreastCancer.csv
├── model.pth
├── requirements.txt
└── README.md
```

* **`DatasetClean.py`** : nettoyage, standardisation et sélection des variables.
* **`NeuralNetworkBC.py`** : définition et entraînement du réseau de neurones.
* **`ModelAnalysis.py`** : évaluation du modèle, calcul des métriques et visualisations.
* **`model.pth`** : paramètres sauvegardés du modèle entraîné.
* **`requirements.txt`** : dépendances Python du projet.

## Installation

Cloner le dépôt, puis installer les dépendances :

```bash
pip install -r requirements.txt
```

Lancer ensuite les scripts depuis le répertoire du projet.

## Modules utilisées

* Python
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

