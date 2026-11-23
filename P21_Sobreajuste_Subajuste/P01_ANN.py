from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# CARGA DE LA INSTANCIA IRIS
# X -> atributos
# y -> clase
iris = load_iris()
X, y = iris.data, iris.target

semilla = 42
n_clases = len(iris.target_names)
proporcion_ruido = 0.25


# DIVISIÓN ESTRATIFICADA 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=semilla,
)


# PARA VALIDACIÓN
X_ajuste, X_val, y_ajuste, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=semilla,
)

conteo_ajuste = Counter(y_ajuste)
conteo_val = Counter(y_val)
conteo_test = Counter(y_test)

print("Especie       Entrenamiento   Validación   Prueba")
for clase, nombre in enumerate(iris.target_names):
    print(
        f"{nombre:<13} {conteo_ajuste[clase]:>13} "
        f"{conteo_val[clase]:>12} {conteo_test[clase]:>8}"
    )


# ESCALADO
# ---> La media y la desviación se calculan con los registros de entrenamiento.
escalador = StandardScaler()
X_ajuste_escalado = escalador.fit_transform(X_ajuste)
X_val_escalado = escalador.transform(X_val)
X_test_escalado = escalador.transform(X_test)


# RUIDO CONTROLADO PARA EL TERCER EXPERIMENTO
# Se copia el arreglo para conservar las etiquetas originales.
# Cada etiqueta elegida se cambia por una de las otras dos especies.
rng = np.random.default_rng(semilla)
y_ajuste_con_ruido = y_ajuste.copy()

n_etiquetas_alteradas = int(len(y_ajuste) * proporcion_ruido)
indices = rng.choice(len(y_ajuste), size=n_etiquetas_alteradas, replace=False)
desplazamientos = rng.integers(1, n_clases, size=n_etiquetas_alteradas)
y_ajuste_con_ruido[indices] = (
    y_ajuste_con_ruido[indices] + desplazamientos
) % n_clases

print(f"\nEtiquetas alteradas en el tercer experimento: "
      f"{n_etiquetas_alteradas} de {len(y_ajuste)}")


# CONSTRUCCIÓN DE LA RED
def crear_modelo(
    capas_ocultas=(5,),
    activacion="relu",
    dropout=0.10,
    tasa_aprendizaje=0.001,
):
    keras.backend.clear_session()
    keras.utils.set_random_seed(semilla)

    modelo = keras.Sequential()
    modelo.add(keras.Input(shape=(X_ajuste.shape[1],)))

    for neuronas in capas_ocultas:
        modelo.add(keras.layers.Dense(neuronas, activation=activacion))
        modelo.add(keras.layers.Dropout(dropout))

    modelo.add(keras.layers.Dense(n_clases, activation="softmax"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return modelo


# Los tres casos usan los mismos atributos y el mismo conjunto de validación
experimentos = [
    ("Subajuste", (1,), 0.0, 5, y_ajuste),
    ("Modelo de referencia", (5,), 0.10, 150, y_ajuste),
    ("Sobreajuste inducido", (64, 64), 0.0, 350, y_ajuste_con_ruido),
]

figura, ejes = plt.subplots(
    2, 3, figsize=(15, 7), sharex="col", layout="constrained"
)

for columna, (nombre, capas, dropout, epocas, etiquetas) in enumerate(experimentos):
    print(f"\n=== {nombre} ===")
    modelo = crear_modelo(capas_ocultas=capas, dropout=dropout)

    # Keras registra pérdida y exactitud en entrenamiento y validación
    # al finalizar cada época. La validación no actualiza los pesos.
    historial = modelo.fit(
        X_ajuste_escalado,
        etiquetas,
        validation_data=(X_val_escalado, y_val),
        epochs=epocas,
        batch_size=16,
        verbose=0,
    ).history

    # La línea vertical marca dónde fue menor la pérdida de validación.
    # Este ejemplo entrena hasta la última época para mostrar toda la curva.
    mejor_epoca = int(np.argmin(historial["val_loss"])) + 1
    eje_epocas = range(1, epocas + 1)

    eje_perdida = ejes[0, columna]
    eje_perdida.plot(eje_epocas, historial["loss"], label="Entrenamiento")
    eje_perdida.plot(eje_epocas, historial["val_loss"], label="Validación")
    eje_perdida.axvline(mejor_epoca, color="gray", linestyle="--")
    eje_perdida.set_title(nombre)
    eje_perdida.set_ylabel("Pérdida")
    eje_perdida.legend()

    eje_exactitud = ejes[1, columna]
    eje_exactitud.plot(eje_epocas, historial["accuracy"], label="Entrenamiento")
    eje_exactitud.plot(eje_epocas, historial["val_accuracy"], label="Validación")
    eje_exactitud.axvline(mejor_epoca, color="gray", linestyle="--")
    eje_exactitud.set_xlabel("Época")
    eje_exactitud.set_ylabel("Exactitud")
    eje_exactitud.set_ylim(0, 1.05)
    eje_exactitud.legend()

    print(f"Época con menor pérdida de validación: {mejor_epoca}")
    print(f"Pérdida de validación mínima: {min(historial['val_loss']):.4f}")
    print(f"Pérdida de validación final: {historial['val_loss'][-1]:.4f}")
    print(f"Exactitud de validación final: {historial['val_accuracy'][-1]:.2%}")

    # Evaluación de los pesos de la última época con los registros de prueba
    probabilidades = modelo.predict(X_test_escalado, verbose=0)
    y_pred = probabilidades.argmax(axis=1)

    aciertos = (y_test == y_pred).sum()
    exactitud = accuracy_score(y_test, y_pred)

    print(f"Aciertos en prueba: {aciertos} de {len(y_test)}")
    print(f"Exactitud en prueba: {exactitud:.2%}")
    print("Matriz de confusión (filas reales, columnas predichas)")
    print("Orden de las clases:", ", ".join(iris.target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_clases)))

plt.show()

# ¿Cómo leer las gráficas?
#  Subajuste: si ambas exactitudes son bajas,
#               la red todavía aprende poco incluso de los ejemplos que ya vio
# Sobreajuste: busca el tramo en que la pérdida de entrenamiento baja y la de validación sube.
#               La exactitud de validación puede quedarse igual si la red sigue eligiendo las mismas clases,
#               pero empieza a asignar más confianza a sus errores.
