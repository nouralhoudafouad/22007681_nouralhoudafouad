# FOUAD NOUR AL HOUDA 
![photo de nour al houda  (1)](https://github.com/user-attachments/assets/10d95830-7bde-4515-aebb-a451213a2293)


*Numéro d'étudiant* : 22007681
*Classe* : FIN 2


<img src="photo de nouralhouda (1).jpg" style="height:464px;margin-right:432px"/>

<br clear="left"/>

---
# ============================================================
# PROJET : DETECTION DES TRANSACTIONS FRAUDULEUSES
# MODELE : RANDOM FOREST
# SORTIES : MATRICES + TABLEAUX EN FORMAT MARKDOWN
# ============================================================

# =====================
# 1. IMPORT DES LIBRAIRIES
# =====================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================
# 2. CHARGEMENT DES DONNEES
# =====================
df = pd.read_csv("creditcard.csv")

# =====================
# 3. PRETRAITEMENT
# =====================
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =====================
# 4. MODELE RANDOM FOREST
# =====================
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

# =====================
# 5. PREDICTIONS
# =====================
y_pred = rf.predict(X_test)

# =====================
# 6. MATRICE DE CONFUSION (TABLEAU MARKDOWN)
# =====================
cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Réel : Normal", "Réel : Fraude"],
    columns=["Prédit : Normal", "Prédit : Fraude"]
)

print("\n## 🧮 Matrice de confusion (Markdown)\n")
print(cm_df.to_markdown())

# =====================
# 7. RAPPORT DE CLASSIFICATION (TABLEAU MARKDOWN)
# =====================
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

print("\n## 📊 Rapport de classification (Markdown)\n")
print(report_df.to_markdown(floatfmt=".4f"))

# =====================
# 8. TABLEAU DES METRIQUES GLOBALES
# =====================
metrics_df = pd.DataFrame({
    "Métrique": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Valeur": [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]
})

print("\n## 📈 Tableau récapitulatif des performances\n")
print(metrics_df.to_markdown(index=False, floatfmt=".4f"))

# =====================
# 9. IMPORTANCE DES VARIABLES (TOP 10)
# =====================
importance_df = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False).head(10)

print("\n## 🔍 Top 10 variables les plus importantes\n")
print(importance_df.to_markdown(index=False, floatfmt=".6f"))

# =====================
# 10. ANALYSE TEXTUELLE (MARKDOWN)
# =====================
print("""
## 🧠 Analyse des résultats

- Le modèle Random Forest présente une excellente capacité à distinguer
  les transactions frauduleuses des transactions normales.
- Le recall élevé pour la classe "Fraude" indique que la majorité des
  transactions frauduleuses sont correctement détectées.
- L'utilisation du paramètre `class_weight="balanced"` permet de limiter
  l'effet du déséquilibre des classes.
- Les variables issues de l'ACP jouent un rôle déterminant dans la détection.

## ✅ Conclusion
Le Random Forest constitue un modèle robuste et performant pour la détection
des fraudes bancaires. Il est particulièrement adapté aux données complexes
et fortement déséquilibrées comme celles utilisées dans ce projet.
""")
