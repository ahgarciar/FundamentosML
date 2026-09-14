from P13_IntroduccionRegresion import MinimosCuadrados

def average(vector):
    avg = sum(vector) / len(vector)
    return avg


def m_estimada(vectorX, vectorY):
    Xprom = average(vectorX)
    Yprom = average(vectorY)
    cov = 0
    for i in range(len(vectorX)):
        cov += (vectorX[i] - Xprom) * (vectorY[i] - Yprom)
    var = 0
    for i in range(len(vectorX)):
        var += (vectorX[i] - Xprom) ** 2  # potencia al cuadrado
    m = cov / var
    m = round(m, 2)
    return m  # m_estimada


def b_estimada(Xprom, Yprom, m_estimada):
    b = Yprom - m_estimada * Xprom
    b = round(b, 2)
    return b  # b_estimada


if __name__ == "__main__":
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #lo que yo esperaba que sucediera
    Yesperada =  [18, 24, 24, 31, 27, 29, 35, 30, 36]
    #lo que realmente paso....
    Yobservada = [20, 25, 22, 30, 25, 27, 32, 28, 40]

    Xavg = average(X)
    Yavg = average(Yobservada)

    m = m_estimada(X, Yobservada)
    b = b_estimada(Xavg, Yavg, m)

    print(m)
    print(b)

    Y_regresion = []
    for i in range(len(X)):
        aux = m * X[i] + b
        aux = round(aux,2)
        Y_regresion.append(aux)

    error_cuadratico = MinimosCuadrados.minimos_cuadrados(Yobservada, Y_regresion)
    print("error cuadratico: ", error_cuadratico)

    from P13_IntroduccionRegresion import MetricasError as calc

    mse = calc.mse(Yobservada, Y_regresion)
    rmse = calc.rmse(Yobservada, Y_regresion)
    mae = calc.mae(Yobservada, Y_regresion)
    mape = calc.mape(Yobservada, Y_regresion)

    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE: ", mae)
    print("MAPE: ", mape)

    from matplotlib import pyplot as plt
    #plt.plot(X, Yesperada, c="blue")
    plt.plot(X, Y_regresion, c="blue")
    plt.scatter(X, Yobservada, c="black")
    plt.show()
