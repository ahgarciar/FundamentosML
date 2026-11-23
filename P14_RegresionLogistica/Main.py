from sklearn.linear_model import LogisticRegression
X = [
[1, 50],
[2, 55],
[3, 60],
[4, 65],
[4, 80],
[5, 75],
[6, 85],
[7, 90],
[8, 95]
]
y = [0, 0, 0, 0, 1, 1, 1, 1, 1]
modelo = LogisticRegression()
modelo.fit(X, y)
print("Intercepto:", modelo.intercept_)
print("Coeficientes:", modelo.coef_)

estudiantes = [
[3, 90],
[7, 60],
[5, 80]
]
print(modelo.predict(estudiantes))
print(modelo.predict_proba(estudiantes))