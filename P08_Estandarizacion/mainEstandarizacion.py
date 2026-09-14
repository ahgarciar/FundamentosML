from sklearn.preprocessing import StandardScaler
import numpy as np
##################################################
registro1 = [10, 8, 9]
registro2 = [9, 10, 7]
registro3 = [7,  7, 8]
instancia = [registro1, registro2, registro3]
instancia = np.array(instancia)
print(instancia)
#######################################################
#scaler = StandardScaler().fit(instancia)
#instancia = scaler.transform(instancia)
scaler = StandardScaler()
instancia = scaler.fit_transform(instancia)
mean = scaler.mean_
print(mean)
var = scaler.var_
print(var)
std = scaler.scale_
print(std)
print(instancia)