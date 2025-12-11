# FOUAD NOUR AL HOUDA 
![photo de nour al houda  (1)](https://github.com/user-attachments/assets/10d95830-7bde-4515-aebb-a451213a2293)


*Numéro d'étudiant* : 22007681
*Classe* : FIN 2


<img src="photo de nouralhouda (1).jpg" style="height:464px;margin-right:432px"/>

<br clear="left"/>

---
# Compte Rendu : Détection de Fraude par Carte de Crédit avec Modèles Prédictifs

## 1. Contexte Métier

### 1.1 Problématique
La fraude par carte de crédit représente un enjeu majeur pour les institutions financières et les consommateurs. Les pertes financières, la confiance des clients et la réputation des entreprises sont en jeu. La détection automatisée de transactions frauduleuses en temps réel est devenue une nécessité.

### 1.2 Objectifs Business
- **Détecter les transactions frauduleuses** avant qu'elles ne soient finalisées
- **Minimiser les faux positifs** pour ne pas bloquer les transactions légitimes
- **Maximiser les vrais positifs** pour capturer le maximum de fraudes
- **Automatiser le processus** de détection pour réduire les coûts opérationnels

### 1.3 Enjeux
- **Déséquilibre des classes** : Les fraudes représentent une infime minorité des transactions
- **Coût d'erreur asymétrique** : Manquer une fraude coûte plus cher qu'un faux positif
- **Confidentialité des données** : Les features sont anonymisées (V1-V28) via PCA pour protéger les informations sensibles

## 2. Code et Architecture Technique

### 2.1 Environnement de Développement
Le notebook est développé sur **Google Colab** avec intégration Kaggle pour le chargement des données.

### 2.2 Bibliothèques Utilisées

**Manipulation de données :**
```python
import pandas as pd
import numpy as np
```

**Visualisation :**
```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly (graph_objs, figure_factory)
```

**Machine Learning :**
```python
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
```

### 2.3 Configuration des Paramètres

```python
# Métriques et paramètres
RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = 4
RANDOM_STATE = 2018

# Splits
VALID_SIZE = 0.20  # 20% validation
TEST_SIZE = 0.20   # 20% test

# Cross-validation
NUMBER_KFOLDS = 5

# LightGBM
MAX_ROUNDS = 1000
EARLY_STOP = 50
VERBOSE_EVAL = 50
```

### 2.4 Chargement des Données

```python
import kagglehub
mlg_ulb_creditcardfraud_path = kagglehub.dataset_download('mlg-ulb/creditcardfraud')
data_df = pd.read_csv(PATH+"/creditcard.csv")
```

## 3. Data Wrangling

### 3.1 Inspection Initiale des Données

**Dimensions du dataset :**
- **Lignes** : 284,807 transactions
- **Colonnes** : 31 variables

**Structure des variables :**
- `Time` : Temps écoulé depuis la première transaction (en secondes)
- `V1 à V28` : Features résultant d'une transformation PCA (anonymisées)
- `Amount` : Montant de la transaction
- `Class` : Variable cible (0 = Non-fraude, 1 = Fraude)

### 3.2 Vérification de la Qualité des Données

```python
# Recherche de valeurs manquantes
total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100)
```

**Résultat** : Aucune valeur manquante détectée dans le dataset - données complètes et prêtes à l'emploi.

### 3.3 Feature Engineering

**Création de la variable temporelle :**
```python
data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))
```

Cette transformation convertit les secondes en heures pour analyser les patterns temporels des fraudes.

### 3.4 Agrégations pour l'Analyse

```python
tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(
    ['min', 'max', 'count', 'sum', 'mean', 'median', 'var']
).reset_index()
```

Création d'un dataframe agrégé par heure et par classe pour identifier les patterns temporels.

## 4. Analyse Exploratoire des Données (EDA)

### 4.1 Distribution des Classes

**Déséquilibre constaté :**
- **Classe 0 (Non-fraude)** : Majorité écrasante (~99.83%)
- **Classe 1 (Fraude)** : Minorité extrême (~0.17%)

**Visualisation** : Graphique à barres montrant le déséquilibre critique qui nécessitera des techniques spéciales (pondération, échantillonnage).

### 4.2 Analyse Temporelle

**Distribution du temps par classe :**
- Density plots comparant les distributions temporelles des fraudes vs non-fraudes
- Identification des heures à risque élevé

**Patterns horaires identifiés :**
- **Montant total des transactions** par heure et par classe
- **Nombre de transactions** par heure
- **Montant moyen** des transactions
- **Valeurs extrêmes** (maximum, minimum, médiane)

**Insight** : Les fraudes peuvent présenter des patterns temporels distincts (heures creuses, nuit, etc.).

### 4.3 Analyse du Montant des Transactions

**Boxplots comparatifs :**
- Avec outliers : Montre l'étendue complète des montants
- Sans outliers : Révèle la distribution centrale

**Statistiques descriptives pour les fraudes :**
```python
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_1.describe()
```

**Visualisation spécifique :** Scatter plot des montants frauduleux en fonction du temps pour identifier des patterns.

### 4.4 Corrélation et Relations entre Variables

**Heatmap de corrélation (Pearson) :**
- Visualisation 14x14 montrant les corrélations entre toutes les features
- Identification des variables fortement corrélées
- Aide à comprendre les relations entre variables PCA

**Relations spécifiques analysées :**
- `V20` vs `Amount` par classe
- `V7` vs `Amount` par classe
- `V2` vs `Amount` par classe
- `V5` vs `Amount` par classe

**Insight** : Certaines variables PCA montrent des relations distinctes avec le montant selon la classe.

### 4.5 Distribution des Features (KDE Plots)

**Analyse complète** : 32 graphiques de densité (8x4) comparant la distribution de chaque variable entre les deux classes.

```python
for feature in var:
    sns.kdeplot(t0[feature], bw=0.5, label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5, label="Class = 1")
```

**Objectif** : Identifier les features qui séparent le mieux les fraudes des non-fraudes visuellement.

## 5. Split des Données

### 5.1 Stratégie de Découpage

**Découpage en trois ensembles :**

```python
# Split 1 : Train+Valid vs Test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, 
    test_size=TEST_SIZE,  # 20%
    random_state=RANDOM_STATE,
    stratify=y  # Maintien du ratio de classes
)

# Split 2 : Train vs Valid
adjusted_valid_size = VALID_SIZE / (1 - TEST_SIZE)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_val, y_train_val,
    test_size=adjusted_valid_size,
    random_state=RANDOM_STATE,
    stratify=y_train_val
)
```

### 5.2 Répartition Finale

- **Train** : ~64% des données (pour l'entraînement)
- **Validation** : ~16% des données (pour le tuning)
- **Test** : ~20% des données (pour l'évaluation finale)

### 5.3 Stratification

**Importance critique** : L'argument `stratify=y` garantit que le ratio fraudes/non-fraudes est préservé dans chaque ensemble, essentiel vu le déséquilibre extrême.

### 5.4 Reconstruction des DataFrames

```python
train_df = pd.DataFrame(X_train, columns=predictors)
train_df[target] = y_train.values

valid_df = pd.DataFrame(X_valid, columns=predictors)
valid_df[target] = y_valid.values

test_df = pd.DataFrame(X_test, columns=predictors)
test_df[target] = y_test.values
```

**Raison** : Le code ultérieur s'attend à manipuler des DataFrames plutôt que des arrays NumPy.

## 6. Algorithme Random Forest

### 6.1 Principe du Random Forest

Le **Random Forest** (Forêt Aléatoire) est un algorithme d'ensemble qui :
- Construit plusieurs arbres de décision indépendants
- Introduit de l'aléatoire dans la sélection des features et des échantillons
- Agrège les prédictions par vote majoritaire (classification)
- Réduit le surapprentissage par rapport à un seul arbre

### 6.2 Configuration du Modèle

```python
clf = RandomForestClassifier(
    n_jobs=NO_JOBS,              # 4 cœurs en parallèle
    random_state=RANDOM_STATE,   # 2018 pour reproductibilité
    criterion=RFC_METRIC,        # 'gini' pour mesurer la qualité du split
    n_estimators=NUM_ESTIMATORS, # 100 arbres dans la forêt
    verbose=False
)
```

### 6.3 Variables Prédictives

```python
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 
              'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 
              'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 
              'V28', 'Amount']
target = 'Class'
```

**Total** : 30 features prédictives pour 1 variable cible binaire.

### 6.4 Entraînement

```python
clf.fit(train_df[predictors], train_df[target].values)
```

Le modèle apprend sur l'ensemble d'entraînement (~64% des données).

### 6.5 Prédiction

```python
preds = clf.predict(valid_df[predictors])
```

Génération des prédictions sur l'ensemble de validation pour évaluation.

### 6.6 Importance des Features

Le Random Forest calcule automatiquement l'importance de chaque variable :

```python
tmp = pd.DataFrame({
    'Feature': predictors, 
    'Feature importance': clf.feature_importances_
})
tmp = tmp.sort_values(by='Feature importance', ascending=False)
```

**Visualisation** : Barplot des 30 features triées par importance décroissante.

**Interprétation** :
- Les features avec haute importance contribuent davantage aux décisions
- Permet d'identifier les variables les plus discriminantes pour la fraude
- Potentiel de réduction de dimensionnalité (éliminer features peu importantes)

## 7. Évaluation du Modèle Random Forest

### 7.1 Matrice de Confusion

```python
cm = pd.crosstab(
    valid_df[target].values, 
    preds, 
    rownames=['Actual'], 
    colnames=['Predicted']
)
```

**Visualisation** : Heatmap 2x2 avec annotations.

**Structure de la matrice** :

|                    | Predicted: Not Fraud | Predicted: Fraud |
|--------------------|----------------------|------------------|
| **Actual: Not Fraud** | Vrais Négatifs (VN)  | Faux Positifs (FP) |
| **Actual: Fraud**     | Faux Négatifs (FN)   | Vrais Positifs (VP) |

**Interprétation métier** :
- **VN** : Transactions légitimes correctement identifiées ✓
- **VP** : Fraudes correctement détectées ✓
- **FP** : Transactions légitimes bloquées par erreur (inconvénient client)
- **FN** : Fraudes non détectées (perte financière critique)

### 7.2 Métriques Dérivées de la Matrice

Bien que non explicitement calculées dans le code, les métriques importantes sont :

**Précision (Precision)** : VP / (VP + FP)
- Quelle proportion des transactions signalées comme fraude sont réellement frauduleuses ?

**Rappel (Recall/Sensitivity)** : VP / (VP + FN)
- Quelle proportion des fraudes réelles sont détectées ?

**Spécificité** : VN / (VN + FP)
- Quelle proportion des transactions légitimes sont correctement identifiées ?

**F1-Score** : 2 × (Precision × Recall) / (Precision + Recall)
- Moyenne harmonique entre précision et rappel

### 7.3 Score ROC-AUC

```python
roc_auc_score(valid_df[target].values, preds)
```

**Signification du ROC-AUC** :
- **ROC** : Receiver Operating Characteristic (courbe)
- **AUC** : Area Under the Curve (aire sous la courbe)
- **Plage** : 0 à 1
  - 0.5 = Performance aléatoire
  - 1.0 = Performance parfaite
  - >0.9 = Excellent
  - 0.8-0.9 = Très bon
  - 0.7-0.8 = Bon

**Avantage** : Métrique robuste au déséquilibre des classes, mesure la capacité du modèle à discriminer entre les classes indépendamment du seuil de décision.

**Interprétation métier** : Un AUC élevé signifie que le modèle classe correctement une transaction frauduleuse au-dessus d'une transaction légitime dans la majorité des cas.

### 7.4 Points Forts de l'Évaluation

1. **Validation séparée** : Utilisation d'un ensemble de validation non vu pendant l'entraînement
2. **Visualisation claire** : Matrice de confusion facilement interprétable
3. **Métrique adaptée** : ROC-AUC pertinent pour les classes déséquilibrées
4. **Importance des features** : Identification des variables clés pour la prédiction

### 7.5 Limites et Améliorations Possibles

**Limites identifiées** :
- Pas de calcul explicite de Precision, Recall, F1-Score
- Pas de courbe ROC visualisée
- Pas d'analyse du seuil de décision optimal
- Pas de prise en compte des coûts métier (coût FN >> coût FP)

**Améliorations recommandées** :
- Implémenter la pondération des classes (`class_weight='balanced'`)
- Ajuster le seuil de décision selon les coûts métier
- Utiliser des techniques de rééchantillonnage (SMOTE, undersampling)
- Comparer avec d'autres algorithmes (XGBoost, LightGBM déjà dans le notebook)
- Validation croisée pour robustesse (K-Fold implémenté plus loin)

---

## Conclusion

Ce projet démontre une approche méthodique de détection de fraude :
- **Compréhension métier** solide du problème
- **Préparation rigoureuse** des données
- **Exploration approfondie** via visualisations
- **Modélisation appropriée** avec Random Forest
- **Évaluation pertinente** avec métriques adaptées

Le Random Forest constitue un bon point de départ, avec ses avantages (interprétabilité via importance des features, robustesse) tout en ouvrant la voie à des modèles plus sophistiqués testés dans la suite du notebook (XGBoost, LightGBM, CatBoost).
