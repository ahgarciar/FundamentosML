import pandas as pd

instancia = pd.read_csv("../InstanciaTennis.csv")

#one hot vector
instancia = pd.get_dummies(instancia,
                           columns=["Outlook", "Temperature", "Humidity", "Wind"]
                           ,dtype=int
                           )

print(instancia.head(5))

instancia = instancia.drop("Day", axis=1)
#drop level --- drop axis

nombre_columnas = list(instancia.columns) #devuelve el nombre de las columnas (encabezados)

nombre_columnas.pop(nombre_columnas.index("Play Tennis"))
nombre_columnas.insert(len(nombre_columnas), "Play Tennis")
#print(nombre_columnas)

instancia = instancia[nombre_columnas]

#instancia.to_csv("../InstanciaTennisCodOneHotVector.csv", index=None)

print()



