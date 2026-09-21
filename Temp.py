X_train = []
y_train = []
X_test = []
y_test = []
pred = []


from sklearn.naive_bayes import GaussianNB
modelo_nb = GaussianNB()
modelo_nb.fit(X_train, y_train)
pred_nb = modelo_nb.predict(X_test)


from sklearn.tree import DecisionTreeClassifier
modelo_id3 = DecisionTreeClassifier(criterion="entropy", random_state=42)
modelo_id3.fit(X_train, y_train)
pred_id3 = modelo_id3.predict(X_test)



from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred, digits=4))



