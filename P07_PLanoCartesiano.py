
import random as rnd

rnd.seed(10)

X = []
Y = []
clase = [] #A clase
for i in range(0, 50, 1):
    X.append(rnd.randint(0, 10) * -1)
    Y.append(rnd.randint(0, 10))
    clase.append(0)

for i in range(0, 50, 1):
    X.append(rnd.randint(0, 10))
    Y.append(rnd.randint(0, 10) * -1)
    clase.append(1)

#print(X)
#print(Y)
#print(clase)

archivo = open("datos_ventas.csv", "w")
for i in range(len(X)):
    archivo.write(str(X[i]) + "," +
                  str(Y[i]) + "," +
                  str(clase[i]) + "\n"
                  )

archivo.close()



from matplotlib import pyplot as plt

plt.scatter(X, Y, c=clase)
plt.show()

