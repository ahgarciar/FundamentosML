X_train = []
y_train = []
X_test = []


from sklearn.naive_bayes import GaussianNB
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)
pred_nb = modelo_nb.predict(X_test)




