import numpy as np
import pandas as pd

instancia = pd.read_csv("../datos_ventas.csv", header=None)
#print(instancia)

X = instancia.iloc[:,:-1]
Y = pd.DataFrame(instancia.iloc[:,-1])
print(X)

Xarray = X.to_numpy()

#from sklearn.preprocessing import StandardScaler as escalador
from sklearn.preprocessing import MinMaxScaler as escalador

scaler = escalador()

Xstd = scaler.fit_transform(Xarray)

Xstd = pd.DataFrame(data=Xstd, columns=[X.columns])

Xstd["class"] = Y

#pendiente de explicar a detalle!!
#outliers = Xstd[Xstd[0] > 2]

#Xstd.to_csv("../datos_ventas_estandarizada.csv", index=False)

Xstd.to_csv("../datos_ventas_normalizada.csv", index=False)

print()


