
import pandas as pd
import numpy as np

datos = pd.read_csv('advertising.csv')

X = datos[['auxilar','TV', 'Radio', 'Newspaper']].to_numpy()
Y = datos['Sales'].to_numpy()

b = X.T.dot(X)
b = np.linalg.inv(b)
b = b.dot(X.T)
b = b.dot(Y)

#print(X)

#print(Y)

print(b)


Y_pred = X.dot(b)

error = 0
for i in range(Y_pred.shape[0]):
    error = (Y[i]   - Y_pred[i])** 2
    print( str(Y[i]) + "\t\t" + str(round(Y_pred[i],1)))
print()

error /= Y_pred.shape[0]

print("RMSE: " + str(error**(1/2)))