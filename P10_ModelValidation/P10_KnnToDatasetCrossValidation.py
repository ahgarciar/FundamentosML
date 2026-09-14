import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("../datos_ventas.csv")

X = dataset.drop(columns=["class"])
y = dataset["class"]

modelo = KNeighborsClassifier(
    n_neighbors=12 ##heuristica...
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    modelo,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("Accuracy por fold:", scores)
print("Accuracy promedio:", scores.mean())
print("Desviación estándar:", scores.std())
