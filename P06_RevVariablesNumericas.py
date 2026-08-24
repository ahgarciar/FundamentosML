import matplotlib.pyplot as plt
import pandas as pandita
import numpy as np

archivo = pandita.read_csv("BookInstancia_Alumnos.csv")

print(archivo)

genero_label = ["f", "m"]

mujeres = archivo[archivo["Genero"] == "f"]
hombres = archivo[archivo["Genero"] == "m"]

datos = [mujeres, hombres]

colums = ["Edad", "Peso", "Altura"]

for index, genero in enumerate(datos):
    print("Genero: " + genero_label[index])
    print("Metricas estadísticas: ")
    for col in colums:
        print("Columna analizada: " + col)
        min = genero[col].min()
        max = genero[col].max()
        promedio = np.mean(genero[col])
        desvstd = np.std(genero[col])
        print("Min: " + str(min))
        print("Max: " + str(max))
        print("Promedio: " + str(promedio))
        print("Desvstd: " + str(desvstd))
        print()

        genero.boxplot(column=col)
        plt.show()




