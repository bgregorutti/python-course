# Utilisation de la librairie Scikit-Learn pour la data science et l'apprentissage automatique

Souvent, les opérations statistiques peuvent s'implémenter en Python sans avoir besoin de librairie supplémentaires. Dans certains cas, il peut être intéressant d'y avoir recours, en particulier pour la contruction de modèles.

Dans **scikit-learn** (créée par l'INRIA en France), de nombreuses classes suivent une interface uniforme avec des méthodes standards comme `fit()`, `transform()`, et `predict()`. Ces méthodes sont conçues pour simplifier l'utilisation des différents algorithmes de machine learning, que ce soit pour l'entraînement, la transformation des données ou les prédictions. Voici un aperçu générique de leur fonctionnement :

### 1. **fit()**
La méthode `fit()` est utilisée pour **apprendre** à partir des données d'entraînement. Elle adapte l'algorithme aux données en trouvant les paramètres optimaux pour un modèle particulier (par exemple, les coefficients dans une régression linéaire).

- **Entrée** : Les données d'entraînement (`X`, et souvent `y` pour les modèles supervisés).
- **Sortie** : Cette méthode ne retourne généralement rien, mais elle met à jour l'état interne de l'objet.

Exemple :
```python
model.fit(X_train, y_train)  # Entraîne le modèle sur les données X_train et y_train
```

### 2. **transform()**
La méthode `transform()` est utilisée pour **modifier** les données en fonction du modèle ou de la transformation qui a été appris avec `fit()`. Elle est souvent utilisée dans le cadre de la **préparation des données** (par exemple, la normalisation, réduction de dimension).

- **Entrée** : Données que l'on souhaite transformer (`X`).
- **Sortie** : Données transformées.

Exemple :
```python
X_scaled = scaler.transform(X_test)  # Applique la transformation (ex : normalisation) aux données de test
```

### 3. **fit_transform()**
`fit_transform()` est une combinaison de `fit()` et `transform()`. C'est une méthode pratique qui permet d'apprendre la transformation des données et de l'appliquer immédiatement en une seule étape. Elle est couramment utilisée pour des transformations de données comme la standardisation ou la réduction de dimension.

- **Entrée** : Données d'entraînement (`X`).
- **Sortie** : Données transformées.

Exemple :
```python
X_scaled = scaler.fit_transform(X_train)  # Normalise et ajuste l'échelle des données en une seule étape
```

### 4. **predict()**
La méthode `predict()` est utilisée pour **prédire** la sortie d'un modèle entraîné sur de nouvelles données. Elle est utilisée dans les modèles supervisés (régression, classification).

- **Entrée** : Nouvelles données (`X`).
- **Sortie** : Prédictions faites par le modèle (`y_pred`).

Exemple :
```python
y_pred = model.predict(X_test)  # Prédit les étiquettes des données de test
```

### 5. **predict_proba()**
Dans certains algorithmes de classification, il est possible d'obtenir les **probabilités** des classes prédites. La méthode `predict_proba()` retourne ces probabilités au lieu des étiquettes finales.

- **Entrée** : Nouvelles données (`X`).
- **Sortie** : Probabilités associées à chaque classe pour chaque donnée.

Exemple :
```python
y_proba = model.predict_proba(X_test)  # Retourne les probabilités des classes
```

### 6. **score()**
La méthode `score()` est utilisée pour **évaluer** les performances d'un modèle sur un ensemble de données en fonction d'une métrique intégrée (comme l'exactitude pour les modèles de classification, ou la variance expliquée pour les modèles de régression).

- **Entrée** : Données d'entrée (`X`) et éventuellement les labels vrais (`y`).
- **Sortie** : Une métrique d'évaluation (ex : précision, score R²).

Exemple :
```python
accuracy = model.score(X_test, y_test)  # Renvoie le score de précision sur les données de test
```