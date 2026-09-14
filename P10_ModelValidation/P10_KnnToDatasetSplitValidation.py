import pandas as pd
import numpy as np

df = pd.read_csv("../datos_ventas.csv")

instancia  = df.to_numpy()

import random as rnd
rnd.shuffle(instancia) ##

entrenamiento_entradas = instancia[:120, :-1]
entrenamiento_salidas = instancia[:120, -1]

prueba_entradas = instancia[120:, :-1]
prueba_salidas = instancia[120:, -1]

# KNN = K vecinos más cercanos
#scikit-learn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=12) # K = 5

neigh.fit(entrenamiento_entradas, entrenamiento_salidas)

correctas  = 0
for i in range(len(prueba_entradas)):
    newCase = [prueba_entradas[i]]
    predicted_class = neigh.predict(newCase)
    print(predicted_class)

    if predicted_class == entrenamiento_salidas[i]:
        correctas = correctas + 1

print("Rendimiento: ")
print(correctas/len(prueba_entradas))

