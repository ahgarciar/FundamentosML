def mse(Yreales, Yestimadas):
    error_cuadratico = 0
    for i in range(len(Yreales)):
        error_cuadratico += (Yreales[i] - Yestimadas[i]) ** 2
    error_cuadratico = error_cuadratico / len(Yreales)
    return error_cuadratico


def rmse(Yreales, Yestimadas):
    error = mse(Yreales, Yestimadas)
    error = error ** (1 / 2)
    return error

def mae(Yreales, Yestimadas):
    error = 0
    for i in range(len(Yreales)):
        error += abs(Yreales[i] - Yestimadas[i])
    error = error / len(Yreales)
    return error

def mape(Yreales, Yestimadas):
    error = 0
    for i in range(len(Yreales)):
        error = Yreales[i] - Yestimadas[i]
        error += abs(error/Yreales[i])
    error = error * 100 / len(Yreales)
    return error
