from collections import Counter

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

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

# PARÁMETROS DE LA RED NEURONAL
capas_ocultas = (5,)        # Una capa oculta con 5 neuronas
activacion = "relu"
dropout = 0.10              # Desactiva aleatoriamente el 10 % durante el entrenamiento
tasa_aprendizaje = 0.001    # Controla el tamaño de los ajustes a los pesos
tamano_lote = 16            # Registros procesados antes de actualizar los pesos
epocas = 150                # Ejecuciones completas por los datos de entrenamiento
semilla_red = 42

keras.utils.set_random_seed(semilla_red)

# ESCALADO DE LOS DATOS
escalador = StandardScaler()
X_train_escalado = escalador.fit_transform(X_train)
X_test_escalado = escalador.transform(X_test)

# MODELO DE RED NEURONAL
modelo = keras.Sequential()
modelo.add(keras.Input(shape=(X_train.shape[1],)))

for neuronas in capas_ocultas:
    modelo.add(keras.layers.Dense(neuronas, activation=activacion))
    modelo.add(keras.layers.Dropout(dropout))

modelo.add(keras.layers.Dense(len(iris.target_names), activation="softmax"))

modelo.compile(
    optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
    loss="sparse_categorical_crossentropy"
)

# ENTRENAMIENTO
modelo.fit(
    X_train_escalado,
    y_train,
    epochs=epocas,
    batch_size=tamano_lote,
    verbose=1
)

# PRUEBA DEL MODELO
probabilidades = modelo.predict(X_test_escalado, verbose=0)
y_pred = probabilidades.argmax(axis=1)

# CALCULO DE LAS METRICAS DEL MODELO
aciertos = (y_test == y_pred).sum()
exactitud = accuracy_score(y_test, y_pred)

print(f"\nAciertos: {aciertos} de {len(y_test)}")
print(f"Exactitud (accuracy): {exactitud:.2%}")

print("\nMatriz de confusión (filas reales, columnas predichas)")
print("Orden de las clases:", ", ".join(iris.target_names))
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))