from collections import Counter

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import (
    ParameterGrid,
    ParameterSampler,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# CONFIGURACIÓN DE LA BÚSQUEDA
tipo_busqueda = "ambas"  # "grid", "random" o "ambas"
n_pruebas_random = 8
semilla_particion = 42
semilla_busqueda = 42
semilla_red = 42

parametros = {
    "capas_ocultas": [(5,), (8, 4)],
    "activacion": ["relu", "tanh"],
    "dropout": [0.0, 0.10],
    "tasa_aprendizaje": [0.001, 0.01],
    "tamano_lote": [16, 32],
    "epocas": [75, 150],
}


# CARGA DE LA INSTANCIA IRIS
iris = load_iris()
X, y = iris.data, iris.target


# DIVISIÓN ESTRATIFICADA 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=semilla_particion,
)


# COMPROBACIÓN DEL BALANCE POR CLASE
conteo_train = Counter(y_train)
conteo_test = Counter(y_test)

print("Especie       Entrenamiento   Prueba")
for clase, nombre in enumerate(iris.target_names):
    print(f"{nombre:<13} {conteo_train[clase]:>13} {conteo_test[clase]:>8}")


# VALIDACIÓN PARA ELEGIR LOS HIPERPARÁMETROS
X_ajuste, X_val, y_ajuste, y_val = train_test_split(
    X_train, y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=semilla_particion,
)

escalador_busqueda = StandardScaler()
X_ajuste_escalado = escalador_busqueda.fit_transform(X_ajuste)
X_val_escalado = escalador_busqueda.transform(X_val)


# CONSTRUCCIÓN Y ENTRENAMIENTO DE LA RED
def entrenar_red(
    X_datos,
    y_datos,
    capas_ocultas=(5,),
    activacion="relu",
    dropout=0.10,
    tasa_aprendizaje=0.001,
    tamano_lote=16,
    epocas=150,
):
    keras.backend.clear_session()
    keras.utils.set_random_seed(semilla_red)

    modelo = keras.Sequential()
    modelo.add(keras.Input(shape=(X_datos.shape[1],)))

    for neuronas in capas_ocultas:
        modelo.add(keras.layers.Dense(neuronas, activation=activacion))
        modelo.add(keras.layers.Dropout(dropout))

    modelo.add(keras.layers.Dense(len(iris.target_names), activation="softmax"))

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tasa_aprendizaje),
        loss="sparse_categorical_crossentropy",
    )

    modelo.fit(
        X_datos,
        y_datos,
        epochs=epocas,
        batch_size=tamano_lote,
        verbose=0,
    )

    return modelo


# SELECCIÓN DE LAS BÚSQUEDAS
if tipo_busqueda not in ("grid", "random", "ambas"):
    raise ValueError('Usa "grid", "random" o "ambas".')

busquedas = {}

if tipo_busqueda in ("grid", "ambas"):
    busquedas["Grid Search"] = ParameterGrid(parametros)

if tipo_busqueda in ("random", "ambas"):
    busquedas["Random Search"] = ParameterSampler(
        parametros,
        n_iter=n_pruebas_random,
        random_state=semilla_busqueda,
    )


# ESCALADO PARA EL ENTRENAMIENTO FINAL
escalador_final = StandardScaler()
X_train_escalado = escalador_final.fit_transform(X_train)
X_test_escalado = escalador_final.transform(X_test)


# BÚSQUEDA, REENTRENAMIENTO Y EVALUACIÓN
for nombre_busqueda, candidatos in busquedas.items():
    print(f"\n=== {nombre_busqueda} ===")

    mejor_exactitud_val = -1.0
    mejores_parametros = None

    for numero, combinacion in enumerate(candidatos, start=1):
        modelo = entrenar_red(X_ajuste_escalado, y_ajuste, **combinacion)

        probabilidades_val = modelo.predict(X_val_escalado, verbose=0)
        y_pred_val = probabilidades_val.argmax(axis=1)
        exactitud_val = accuracy_score(y_val, y_pred_val)

        print(f"Combinación {numero}/{len(candidatos)} | "
              f"Exactitud de validación: {exactitud_val:.2%}")

        if exactitud_val > mejor_exactitud_val:
            mejor_exactitud_val = exactitud_val
            mejores_parametros = combinacion

    print(f"\nMejor exactitud de validación: {mejor_exactitud_val:.2%}")
    print("Mejores parámetros:")
    for parametro, valor in mejores_parametros.items():
        print(f"  {parametro} = {valor}")

    # REENTRENAMIENTO
    modelo_final = entrenar_red(
        X_train_escalado, y_train, **mejores_parametros
    )

    # PRUEBA FINAL
    probabilidades = modelo_final.predict(X_test_escalado, verbose=0)
    y_pred = probabilidades.argmax(axis=1)

    aciertos = (y_test == y_pred).sum()
    exactitud = accuracy_score(y_test, y_pred)

    print(f"\nAciertos en prueba: {aciertos} de {len(y_test)}")
    print(f"Exactitud en prueba: {exactitud:.2%}")
    print("\nMatriz de confusión (filas reales, columnas predichas)")
    print("Orden de las clases:", ", ".join(iris.target_names))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))