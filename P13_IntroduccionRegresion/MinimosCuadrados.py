

def minimos_cuadrados(Yreales, Yestimadas):
    error_cuadratico = 0
    for i in range(len(Yreales)):
        error_cuadratico += (Yreales[i]-Yestimadas[i])**2
    return error_cuadratico

