from sklearn.preprocessing import MinMaxScaler
import numpy as np
registro1 = [10, 8, 9]
registro2 = [9, 10, 7]
registro3 = [7,  7, 8]
instancia = [registro1, registro2, registro3]
instancia = np.array(instancia)
print(instancia)
scaler = MinMaxScaler().fit(instancia)
instancia = scaler.transform(instancia)
#scaler = MinMaxScaler()
#instancia = scaler.fit_transform(instancia)
min_value = scaler.data_min_
print(min_value)
max_value = scaler.data_max_
print(max_value)
print(instancia)