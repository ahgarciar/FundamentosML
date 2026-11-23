from collections import Counter

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# CARGA DE LA INSTANCIA
# Instancia: iris - Registros: 150
# X contiene cuatro atributos por registro
# y contiene la especie (clase)
iris = load_iris()
X, y = iris.data, iris.target

# SEPERACION DE LA INSTANCIA CONSERVANDO LA PROPORCION POR CLASE
#DIVISION  80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# COMPROBACION DEL BALANCEO POR CLASE
conteo_train = Counter(y_train)
conteo_test = Counter(y_test)

print("Especie       Entrenamiento   Prueba")
for clase, nombre in enumerate(iris.target_names):
    print(f"{nombre:<13} {conteo_train[clase]:>13} {conteo_test[clase]:>8}")

# PIPELINE. MODELO DE RED NEURONAL
modelo = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(5,),
        solver="lbfgs",
        max_iter=1000,
        random_state=42
    )
)

# ESCALADO Y ENTRENAMIENTO DEL MODELO CON LOS DATOS DE ENTRENAMIENTO
modelo.fit(X_train, y_train)

# PRUEBA DEL MODELO
y_pred = modelo.predict(X_test)

# CALCULO DE LAS METRICAS DEL MODELO
aciertos = (y_test == y_pred).sum()
exactitud = accuracy_score(y_test, y_pred)

print(f"\nAciertos: {aciertos} de {len(y_test)}")
print(f"Exactitud (accuracy): {exactitud:.2%}")

print("\nMatriz de confusión (filas reales, columnas predichas)")
print("Orden de las clases:", ", ".join(iris.target_names))
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))