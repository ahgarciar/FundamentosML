import matplotlib.pyplot as plt
import pandas as pandita
import numpy as np

archivo = pandita.read_csv("BookInstancia_Alumnos.csv")

print(archivo)

#devuelve el nombre de las columnas del dataframe
#columns = archivo.columns
#print(columns)

colums = ["Edad", "Peso", "Altura"]

print("Metricas estadísticas: ")
for col in colums:
    print("Columna analizada: " + col)
    min = archivo[col].min()
    max = archivo[col].max()
    promedio = np.mean(archivo[col])
    desvstd = np.std(archivo[col])
    print("Min: " + str(min))
    print("Max: " + str(max))
    print("Promedio: " + str(promedio))
    print("Desvstd: " + str(desvstd))
    print()

    archivo.boxplot(column=col)
    plt.show()


#df[df["Ciudad"] == "Tampico"]