
# ============================================================
# 0. IMPORTACION DE LIBRERIAS

# numPy => Permite trabajar con arreglos y valores numéricos
# -> np.isfinite() verifica que los atributos sean números finitos
# -> np.nan representa un resultado indefinido cuando todos los modelos empatan en cada ejecución
# -> np.isnan() permite identificar ese caso
import numpy as np

# pandas => Permite leer y manipular datos en forma de tabla
# Se usa para leer el CSV, convertir los atributos a números,
# contar los registros por clase, calcular los rangos y exportar
# las tablas de resultados a archivos CSV
import pandas as pd

# matplotlib => Permite mostrar y personalizar gráficas
# Se usa para colocar títulos y etiquetas en la gráfica de exactitud,
# el diagrama de cajas y la gráfica de rangos promedio
import matplotlib.pyplot as plt

# scikit-posthocs => Proporciona pruebas de comparación por pares
# Se usa para aplicar Nemenyi cuando Friedman detecta una diferencia global
# -> Devuelve un p-valor para cada par de algoritmos
import scikit_posthocs as sp

# sciPy => Incluye funciones para realizar pruebas estadísticas
# friedmanchisquare() aplica la prueba de Friedman a las exactitudes
# emparejadas de la RNA, la regresión logística y KNN
# -> Devuelve el estadístico de la prueba y su p-valor
from scipy.stats import friedmanchisquare

# scikit-learn => Proporciona herramientas para preparar los datos,
# entrenar modelos de clasificación y evaluar sus predicciones
# -> Divide los registros en entrenamiento y prueba
# -> Con stratify=y, procura conservar la proporción de cada clase
from sklearn.model_selection import train_test_split

# Estandariza los atributos usando la media y la desviación estándar
# calculadas con el conjunto de entrenamiento.
# Después aplica esos mismos parámetros a los datos de prueba
from sklearn.preprocessing import StandardScaler

# Calcula la exactitud al dividir el número de aciertos entre
# el total de registros evaluados
from sklearn.metrics import accuracy_score

# Entrena un perceptrón multicapa para clasificar los registros
# -> En este experimento, la red tiene una capa oculta de 10 neuronas
from sklearn.neural_network import MLPClassifier

# Entrena un clasificador de regresión logística
# -> El modelo estima probabilidades de pertenencia a las clases y elige la más probable
from sklearn.linear_model import LogisticRegression

# Clasifica cada registro mediante el voto de sus vecinos más cercanos
# -> En este experimento, se consultan cinco vecinos.
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# 1. CONFIGURACIÓN DEL EXPERIMENTO
ruta_csv = "iris.csv"

# Cada ejecución genera una partición diferente del mismo dataset
# -> Se realizan 30 particiones estratificadas con semillas distintas
# Se eligieron 30 como una cantidad práctica para observar cuánto
# varía la exactitud al cambiar los registros de entrenamiento y prueba.
# -> El teorema del límite central explica por qué, bajo ciertas
# condiciones, la distribución de las medias muestrales se aproxima
# a una normal al aumentar el número de observaciones independientes
n_ejecuciones = 30

# 80% de los registros para entrenamiento y 20 % para la prueba de los modelos de ML
porcentaje_prueba = 0.20

# Umbral para interpretar el p-valor de Friedman y del post hoc
alfa = 0.05

# Cantidad de vecinos que consulta KNN para clasificar un registro.
n_vecinos = 5

# ============================================================
# 2. CARGA Y VALIDACIÓN DEL DATASET
# Se lee el CSV con encabezados.
# Para este trabajo:
# -> La última columna corresponde a la clase
# -> Las clases conservan etiquetas como "setosa" y códigos ordinales como "01"
# -> keep_default_na=False permite conservar etiquetas legítimas como "NA"
# -> Los campos vacíos se revisan explícitamente posteriormente
instancia = pd.read_csv(
    ruta_csv,
    dtype=str,
    keep_default_na=False
)

if instancia.empty:
    raise ValueError("El CSV debe contener al menos un registro")

if instancia.shape[1] < 2:
    raise ValueError(
        "El CSV debe tener al menos un atributo y una columna de clase"
        "Si usa punto y coma como separador, agrega sep=';' a pd.read_csv"
    )

# X contiene los atributos
# -> :-1 -> toma todas las columnas
# hasta la penúltima, conservando todas las filas
X_original = instancia.iloc[:, :-1]

# y contiene una etiqueta por registro
# -> Al seleccionar la última columna con -1 se obtiene una Serie, es decir, un vector de etiquetas
# -> strip() elimina espacios accidentales al inicio y al final
y = instancia.iloc[:, -1].str.strip()

if y.eq("").any():
    raise ValueError(
        f"La columna de clase '{y.name}' contiene etiquetas vacías"
    )

# Se convierte cada columna de atributos a valores numéricos
# errors="coerce" convierte en NaN los valores que no pueden interpretarse como números
#   -> Esto permite indicar qué columna contiene el problema.
X = X_original.apply(pd.to_numeric, errors="coerce")

# Los modelos requieren números finitos.
# La revisión detecta campos vacíos, texto que no representa un número e infinitos.
for columna in X.columns:
    valores_invalidos = ~np.isfinite(X[columna].to_numpy(dtype=float))

    if valores_invalidos.any():
        ejemplos = (
            X_original.loc[valores_invalidos, columna]
            .head(3)
            .tolist()
        )
        raise ValueError(
            f"El atributo '{columna}' contiene valores no numéricos, "
            f"vacíos o infinitos. Ejemplos: {ejemplos!r}"
        )

# value_counts() cuenta cuántos registros corresponden a cada clase
conteo_clases = y.value_counts()
n_clases = len(conteo_clases)

if n_clases < 2:
    raise ValueError("La clasificación requiere al menos dos clases")

# La partición estratificada de scikit-learn requiere al menos
# dos registros por clase para intentar repartirlos entre los conjuntos
if conteo_clases.min() < 2:
    clases_escasas = conteo_clases[conteo_clases < 2].index.tolist()
    raise ValueError(
        "Cada clase debe tener al menos dos registros para estratificar. "
        f"Clases con un solo registro: {clases_escasas}."
    )

print("Registros:", X.shape[0])
print("Atributos:", X.shape[1])
print("Columna de clase:", y.name)
print("Clases:", n_clases)
print("\nREGISTROS POR CLASE")
print(conteo_clases)


# ============================================================
# 3. CREACION DE LISTAS PARA ALMACENAR LAS EXACTITUDES (Accuracy)
# La posición i de las tres listas corresponde a la misma partición
# Esto permite comparar los modelos por pares dentro de cada ejecución
resultados_rna = []
resultados_rl = []
resultados_knn = []

# ============================================================
# 4. EJECUCIONES DEL EXPERIMENTO
for i in range(n_ejecuciones):
    # Las semillas 1, 2, ..., 30 permiten repetir el experimento
    # -> En cada ejecución, los tres modelos reciben exactamente
    # los mismos registros de entrenamiento y de prueba
    semilla = i + 1

    # --------------------------------------------------------
    # 4.1 Split validation estratificada
    # stratify=y procura conservar la proporción de cada clase
    # en los conjuntos de entrenamiento y prueba
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=porcentaje_prueba,
            stratify=y,
            random_state=semilla
        )
    except ValueError as error:
        raise ValueError(
            "No se pudo crear la partición estratificada "
            "Revisa el porcentaje de prueba y la cantidad de registros "
            f"por clase. Detalle: {error}"
        ) from error

    # Con clases muy escasas, la asignación de registros enteros
    # puede dejar alguna clase fuera de uno de los conjuntos.
    if y_train.nunique() < n_clases or y_test.nunique() < n_clases:
        raise ValueError(
            f"En la ejecución {i + 1} faltó al menos una clase en "
            "entrenamiento o prueba. Ajusta la proporción de prueba "
            "o usa un dataset con más registros de las clases escasas"
        )

    # KNN necesita al menos tantos ejemplos de entrenamiento
    # como vecinos se le pide consultar.
    if len(X_train) < n_vecinos:
        raise ValueError(
            f"KNN requiere al menos {n_vecinos} registros de "
            f"entrenamiento. La partición tiene {len(X_train)}"
        )

    # --------------------------------------------------------
    # 4.2 Estandarización
    # StandardScaler calcula la media y la desviación estándar
    # de cada atributo usando exclusivamente el entrenamiento
    # -> Después aplica esa misma transformación al conjunto de prueba
    # -> Así se evita que la información de prueba influya en el ajuste
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --------------------------------------------------------
    # 4.3 Modelo 1. Red neuronal artificial
    # CONFIGURACION DEL MODELO
    # -> Se conserva una capa oculta de 10 neuronas y activación ReLU
    # -> lbfgs es el metodo utilizado para ajustar los pesos de la red
    # -> random_state hace reproducible la inicialización de los pesos
    modelo_rna = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="relu",
        solver="lbfgs",
        max_iter=2000,
        random_state=semilla
    )

    # ENTRENAMIENTO
    # fit aprende de los atributos y las clases de entrenamiento
    modelo_rna.fit(X_train, y_train)

    # PREDICION
    # predict recibe únicamente los atributos de prueba y devuelve
    # una etiqueta predicha por cada registro
    pred_rna = modelo_rna.predict(X_test)

    # EVALUACION
    # La exactitud es la proporción de etiquetas predichas correctamente
    # Por ejemplo, 27 aciertos de 30 registros producen 0.90
    accuracy_rna = accuracy_score(y_test, pred_rna)

    # ALMACENAMIENTO
    resultados_rna.append(accuracy_rna)

    # --------------------------------------------------------
    # 4.4 Modelo 2. Regresión logística
    # CONFIGURACION DEL MODELO
    # La regresión logística estima probabilidades de pertenecer a las clases
    # -> predict devuelve la etiqueta de mayor probabilidad
    # -> Puede trabajar con dos clases o con varias, como en Iris
    # -> C controla la regularización. Un C menor la hace más fuerte
    # -> max_iter fija el máximo de iteraciones del optimizador
    modelo_rl = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000
    )

    # ENTRENAMIENTO
    modelo_rl.fit(X_train, y_train)

    # PREDICION
    pred_rl = modelo_rl.predict(X_test)

    # EVALUACION
    accuracy_rl = accuracy_score(y_test, pred_rl)

    # ALMACENAMIENTO
    resultados_rl.append(accuracy_rl)

    # --------------------------------------------------------
    # 4.5 Modelo 3. K vecinos más cercanos
    # CONFIGURACION DEL MODELO
    # KNN busca los cinco registros de entrenamiento más cercanos según la distancia euclidiana
    # -> Con weights="uniform",
    # cada vecino aporta un voto con el mismo peso
    modelo_knn = KNeighborsClassifier(
        n_neighbors=n_vecinos,
        metric="euclidean",
        weights="uniform"
    )

    # ENTRENAMIENTO
    modelo_knn.fit(X_train, y_train)

    # PREDICION
    pred_knn = modelo_knn.predict(X_test)

    # EVALUACION
    accuracy_knn = accuracy_score(y_test, pred_knn)

    # ALMACENAMIENTO
    resultados_knn.append(accuracy_knn)

    # --------------------------------------------------------
    # 4.6 Resultados de la ejecución
    print(
        f"Ejecución {i + 1:02d} | "
        f"RNA: {accuracy_rna:.4f} | "
        f"RL: {accuracy_rl:.4f} | "
        f"KNN: {accuracy_knn:.4f}"
    )

# ============================================================
# 5. TABLA DE RESULTADOS
# Cada fila representa una partición
# -> Cada columna contiene la exactitud de un modelo en esa misma partición
resultados = pd.DataFrame({
    "RNA": resultados_rna,
    "RL": resultados_rl,
    "KNN": resultados_knn
})

# Se agrega el indice de ejecucion en la tabla de resultados
resultados.index = range(1, n_ejecuciones + 1)
resultados.index.name = "Ejecucion"

# Se redondea la salida en pantalla a 4 decimales
print("\nRESULTADOS POR EJECUCIÓN")
print(resultados.round(4))

# El CSV y los cálculos posteriores conservan los valores originales sin redondeo
resultados.to_csv("resultados_algoritmos.csv")

# ============================================================
# 6. ESTADÍSTICA DESCRIPTIVA
# -> La media y la mediana resumen la exactitud de cada modelo
# -> La desviación estándar muestral describe cuánto varían sus
#       exactitudes entre particiones
# -> Mínimo y máximo dan los extremos
resumen = pd.DataFrame({
    "Media": resultados.mean(),
    "Mediana": resultados.median(),
    "Desviacion": resultados.std(),
    "Minimo": resultados.min(),
    "Maximo": resultados.max()
})

print("\nESTADÍSTICA DESCRIPTIVA")
print(resumen.round(4))

resumen.to_csv("estadistica_descriptiva.csv")

# ============================================================
# 7. RANGOS POR EJECUCIÓN
# -> axis=1 hace que el ordenamiento se realice dentro de cada fila
# -> La exactitud más alta recibe el rango 1
# -> Si dos modelos empatan en el primer lugar, ambos reciben 1.5, que es el promedio de los rangos 1 y 2
# -> Si los tres modelos empatan, los tres reciben 2, que es el promedio de los rangos 1, 2 y 3
# -> Si dos modelos empatan en el segundo lugar, entonces ambos reciben 2.5, que es el promedio de los rangos 2 y 3
rangos = resultados.rank(
    axis=1,
    ascending=False,
    method="average"
)

rangos_promedio = rangos.mean().sort_values()

print("\nRANGOS POR EJECUCIÓN")
print(rangos)

print("\nRANGOS PROMEDIO")
print(rangos_promedio.round(4))

rangos.to_csv("rangos_algoritmos.csv")

# ============================================================
# 8. PRUEBA NO PARAMÉTRICA DE FRIEDMAN
# -> Cada partición actúa como un bloque
# -> Las tres columnas deben estar alineadas porque se comparan resultados de la misma prueba

# H0: No existen diferencias sistematicas entre
#     los resultados de los modelos
# H1: Al menos un modelo presenta diferencias

# El mismo registro puede aparecer en varias particiones
# Por ello, el p-valor se utiliza como parte de un análisis exploratorio
if (resultados.nunique(axis=1) == 1).all():
    # Con empate total en todas las filas, la prueba queda indefinida
    estadistico = np.nan
    p_valor = np.nan
else:
    estadistico, p_valor = friedmanchisquare(
        resultados["RNA"],
        resultados["RL"],
        resultados["KNN"]
    )

print("\nPRUEBA DE FRIEDMAN")
print(f"Nivel de significancia: {alfa}")

if np.isnan(p_valor):
    print("Friedman quedó indefinida por empate total en cada ejecución.")
else:
    print(f"Estadístico: {estadistico:.6f}")
    print(f"P-valor: {p_valor:.6f}")


# ============================================================
# 9. INTERPRETACIÓN Y PRUEBA POST HOC DE NEMENYI
if np.isnan(p_valor):
    print("Se omite el post hoc por el empate total")
elif p_valor < alfa:
    print("\nSe rechaza la hipotesis nula (H0).")
    print("Existen diferencias estadisticamente significativas.")
    print("Se revisan los pares para identificar dónde aparecen diferencias.")

    # 9.1 PRUEBA POST HOC DE NEMENYI
    # -> Friedman da un resultado global
    # -> Nemenyi devuelve una matriz con un p-valor para cada par de algoritmos
    # Se entrega la tabla de exactitudes porque la función
    # calcula internamente los rangos de cada ejecución
    posthoc = sp.posthoc_nemenyi_friedman(resultados)

    print("\nPRUEBA POST HOC DE NEMENYI")
    print(posthoc.round(6))

    posthoc.to_csv("resultados_posthoc.csv")

    # 9.2 COMPARACION POR PARES
    # La matriz es simétrica. Basta con consultar cada par una vez.
    comparaciones = [
        ("RNA", "RL"),
        ("RNA", "KNN"),
        ("RL", "KNN")
    ]

    print("\nCOMPARACIONES POR PARES")

    for algoritmo_a, algoritmo_b in comparaciones:
        p_par = posthoc.loc[algoritmo_a, algoritmo_b]

        print(f"\n{algoritmo_a} vs {algoritmo_b}")
        print(f"P-valor: {p_par:.6f}")

        if p_par < alfa:
            # Se reporta cuál de los dos obtuvo el menor rango
            # promedio, es decir, cuál tendió a quedar mejor ubicado.
            mejor_algoritmo = rangos_promedio.loc[
                [algoritmo_a, algoritmo_b]
            ].idxmin()

            print("Se detectó una diferencia significativa.")
            print("Menor rango promedio:", mejor_algoritmo)
        else:
            print("No se detectó una diferencia significativa.")
else:
    # Un p-valor mayor o igual que alfa indica que la evidencia
    # es insuficiente para rechazar H0 con este experimento.
    print("\nNo se rechaza la hipotesis nula (H0).")
    print("No existe evidencia suficiente de diferencias.")
    print("Se omite la prueba post hoc.")


# ============================================================
# 10. COEFICIENTE W DE KENDALL
# -> Complementa la prueba de Friedman
# -> Permite medir la concordancia de los rangos
# -> W describe cuánto concuerda el ordenamiento de los modelos entre particiones
# -> Valores cercanos a 1 indican mayor acuerdo en el orden de los rangos
# -> W no mide puntos de exactitud.
n = len(resultados)
k = len(resultados.columns)

if not np.isnan(estadistico):
    w = estadistico / (n * (k - 1))

    print("\nCOEFICIENTE W DE KENDALL")
    print(f"W = {w:.6f}")


# ============================================================
# 11. GRÁFICAS
# 11.1 Exactitud por ejecución.
# Cada punto se refiere a una partición específica del dataset.
resultados.plot(marker="o", figsize=(12, 5))

plt.title("Exactitud por ejecución")
plt.xlabel("Ejecución")
plt.ylabel("Exactitud")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 11.2 Diagrama de cajas
# -> Permite comparar las medianas y la dispersión de las exactitudes
resultados.boxplot(figsize=(8, 5))

plt.title("Distribución de la exactitud por algoritmo")
plt.ylabel("Exactitud")
plt.tight_layout()
plt.show()

# 11.3 Rangos promedio
# -> En esta gráfica, una barra más baja representa una mejor posición
rangos_promedio.plot(kind="bar", figsize=(8, 5), rot=0)

plt.title("Rangos promedio de Friedman")
plt.xlabel("Algoritmo")
plt.ylabel("Rango promedio (menor es mejor)")
plt.ylim(0, k + 0.2)
plt.tight_layout()
plt.show()