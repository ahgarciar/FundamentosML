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

#SPLIT VALIDATION
#dividiendo la instancia en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
X, y,
    test_size=0.20, #80% de train / 20% de prueba
    stratify=y, #busca asegurar que se mantenga la proporcion de las clases
    random_state=42
)

#se encarga de entrenar el modelo!!
modelo.fit(X_train, y_train)

#ejecucion del modelo entrenado para ver como se comporta
predicciones = modelo.predict(X_test)

#Calculo de las metricas:

print("Resultados de la prueba:")
for i in range(len(predicciones)):
    print(str(y_test[i]) + "\t" + str(predicciones[i]))
print()

print("Entrenamiento por especie:", np.bincount(y_train))
print("Prueba por especie:", np.bincount(y_test))
print("Exactitud holdout: " +
      str(round(accuracy_score(y_test, predicciones), 2))
      )

