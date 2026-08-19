
import pandas as pandita
import numpy as np

archivo = pandita.read_csv("BookInstancia_Alumnos.csv")

print(archivo)

#devuelve el nombre de las columnas del dataframe
columns = archivo.columns
print(columns)

print("Datos de la columna edad: ")
df_edad = archivo["Edad"]
print(df_edad)

print("Edad que tiene el alumno con menor edad:")
print(min(df_edad))

print("Edad que tiene el alumno con mayor edad:")
print(max(df_edad))

print("Promedio de edad del grupo:")
print(np.mean(df_edad))

print("Desviacion estandar del grupo:")
print(np.std(df_edad))


