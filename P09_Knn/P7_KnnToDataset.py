import math

import pandas as pd
import numpy as np

df = pd.read_csv("../datos_ventas.csv", header=None)

columns = df.columns

entradas = df[columns[:-1]] #selecciona todas las columnas a excepción de la última
salidas = df[columns[-1]] #selecciona solo la clase

entradas = entradas.to_numpy()
salidas = salidas.to_numpy()

print("Entradas:")
print(entradas)
print("Salidas:")
print(salidas)

# KNN = K vecinos más cercanos

k = int(len(entradas) ** (1/2))
#k = math.sqrt(len(entradas))

#scikit-learn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=k) # K = 5

neigh.fit(entradas, salidas)

#Obtiene la clase más probabable (moda) con base en la aplicación del knn
newCase = [[5, 5]]  ##<<<<-----
# -6	7	0
predicted_class = neigh.predict(newCase)
print(predicted_class)

#Calcula las probabilidades de que el individuo pertenezca a cada clase
print(neigh.predict_proba(newCase))


