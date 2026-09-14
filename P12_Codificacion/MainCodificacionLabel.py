import pandas as pd

instancia = pd.read_csv("../InstanciaTennis.csv")

nombre_columnas = list(instancia.columns)

from sklearn.preprocessing import LabelEncoder
encoderOutlook = LabelEncoder()
encoderTemperature = LabelEncoder()
encoderHumidity = LabelEncoder()
encoderWind = LabelEncoder()

instancia["Label_Outlook"] = encoderOutlook.fit_transform(instancia["Outlook"])
instancia["Label_Temperature"] = encoderOutlook.fit_transform(instancia["Temperature"])
instancia["Label_Humidity"] = encoderOutlook.fit_transform(instancia["Humidity"])
instancia["Label_Wind"] = encoderOutlook.fit_transform(instancia["Wind"])

instancia = instancia.drop(["Day","Outlook", "Temperature", "Humidity", "Wind"], axis=1)

nombre_columnas = list(instancia.columns) #devuelve el nombre de las columnas (encabezados)

nombre_columnas.pop(nombre_columnas.index("Play Tennis"))
nombre_columnas.insert(len(nombre_columnas), "Play Tennis")
#print(nombre_columnas)

instancia = instancia[nombre_columnas] #recupera las columnas en el nuevo orden

#instancia.to_csv("../InstanciaTennisCodLabel.csv", index=None)

print()



