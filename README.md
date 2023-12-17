# Détection des fraudes à la carte de crédit

## Introduction du projet

Ce projet consiste à construire un modèle de prédiction anti-fraude des cartes de crédit en utilisant les données historiques des transactions par carte de crédit, afin de détecter à l'avance le vol des cartes de crédit des clients.

## Dataset

Nous utilisont le jeu de données sur la site https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

> The dataset contains transactions made by credit cards in September 2013 by European cardholders.
> 
> This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
> 
> It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Scénarios de données

Ce jeu de données est constitué des transactions par carte de crédit, la problématique est de prédire si le client sera victime de fraude à la carte de crédit. Il y a seulement deux situations : fraude et non fraude. Et comme les données sont déjà classifiés par la colonne "Class", il s'agit d'un scénario d'apprentissage supervisé. C'est la raison pour laquelle la prédiction de fraude à la carte de crédit est une problématique de classification binaire.

## Contributions

Ce projet a été réalisé conjointement par SU Lifang et LI Ningyu.

* Idée du projet : Les deux
* Code :
  - Partie “Logistique Régression” : Ningyu
  - Partie “Random Forest” : LiFang
* Nettoyage des données : Les 2
* Tuning des paramètres  :  
  - Random Forest : RandomizedSearchCV et hyperopt bayesienne : LiFang
  - Logistique Régression : find_best_c  & GridSearchCV : Ningyu
* Algo Random Forest : LiFang
* Algo Logistique Régression : Ningyu
* Graphes : Les deux
