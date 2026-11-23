import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
StratifiedKFold, cross_val_score, train_test_split
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#instancia/conjunto de datos / dataset
iris = load_iris()

#dividir la instancia en atributos y clase
# 4 atributos
# 3 clases
# 150 registros  .... 50 por clase
X, y = iris.data, iris.target

print("Clases:", iris.target_names)
print("Registros por Clase:", np.bincount(y))

#modelo de Machine Learning .... PERCEPTRON MULTICAPA (Red Neuronal)
modelo = (
    make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
        max_iter=2000, random_state=42)
    )
)

#CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for ronda, (idx_ent, idx_val) in enumerate(cv.split(X, y), start=1):
    conteo_ent = np.bincount(y[idx_ent])
    conteo_val = np.bincount(y[idx_val])
    print(f"Ronda {ronda}: ajuste {conteo_ent}, evaluacion {conteo_val}")

exactitudes = cross_val_score(modelo, X, y, cv=cv, scoring="accuracy")

print()
for ronda, exactitud in enumerate(exactitudes, start=1):
    print("Ronda " + str(ronda) + ": " + str(exactitud))

print()
print("Promedio: " + str(exactitudes.mean()))
print("Desviacion estandar: " + str(100 * exactitudes.std()))

