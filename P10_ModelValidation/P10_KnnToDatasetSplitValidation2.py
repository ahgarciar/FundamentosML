import pandas as pd
import numpy as np

#####
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("../datos_ventas.csv")

X = dataset.drop(columns=["class"])
y = dataset["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.20, #20%
    random_state=42,
    stratify=y # para buscar el mayor balance de las clases en las divisiones
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


###########

# KNN = K vecinos más cercanos
#scikit-learn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=12) # K = 5

neigh.fit(X_train, y_train)

correctas  = 0
for i in range(len(X_test)):
    newCase = X_test.iloc[[i]]
    predicted_class = neigh.predict(newCase)
    print(predicted_class)

    if predicted_class == y_test.iloc[i]:
        correctas = correctas + 1

print("Rendimiento: ")
print(correctas/len(X_test))

