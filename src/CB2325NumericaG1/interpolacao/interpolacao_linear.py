import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

def _ordenar_coordenadas(x: list, y: list) -> list:
        x_np = np.array(x)
        y_np = np.array(y)

        idx = np.argsort(x_np)

        x_ord = x_np[idx]
        y_ord = y_np[idx]

        return x_ord, y_ord

def _plotar(x: list, y: list, f: Callable):
    x_points = np.linspace(x[0], x[-1], 100)
    y_points = [f(xp) for xp in x_points]

    _, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x_points, y_points)
    plt.show()

    return

def lin_interp(x: list, y: list) -> Callable:
    x, y = _ordenar_coordenadas(x, y)

    def f(x1):
        if x1 < x[0]:
            return y[0]
        elif x1 > x[-1]:
            return y[-1]
        else:
            for i in range (1, len(x)):
                if x[i] >= x1 >= x[i-1]:
                    a = (y[i] - y[i-1])/(x[i] - x[i-1])
                    b = y[i-1]
                    
                    y1 = b + (x1 - x[i-1]) * a
                    
                    return y1

    _plotar(x, y, f)

    return f

if __name__=='__main__':
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    y = [1, 2, 0, 4, 5, 7, 1, 0]

    p = lin_interp(x,y)
    print(p)
    print(p(1.5))