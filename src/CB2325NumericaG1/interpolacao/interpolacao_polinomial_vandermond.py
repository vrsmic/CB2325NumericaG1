import matplotlib.pyplot as plt
import numpy as np
from typing import Callable

def _ordenar_coordenadas(x: list, y: list) -> list:
    """Ordena as coordenadas mantendo 'pareamento'.
    
    Pega as coordenadas x e y de cada ponto, ordena em ordem crescente
    as coordenadas x e mantém pareamento com y. Função privada, auxiliar
    da função principal lin_interp.

    Args:
        x: lista das coordenadas x, em x[i], de cada ponto i.
        y: lista das coordenadas y, em y[i], de cada ponto i.

    Returns:
        x_ord: lista das coordenadas x em ordem crescente.
        y_ord: lista das coordenadas y, pareadas com as coordenadas x.
    """
    if len(x) != len(y):
        raise RuntimeError(f"x e y devem ter a mesma quantidade de elementos")
    
    x_np = np.array(x)
    y_np = np.array(y)

    idx = np.argsort(x_np)

    x_ord = x_np[idx]
    y_ord = y_np[idx]

    return x_ord, y_ord

def _plotar(x: list,
            y: list,
            f: Callable,
            titulo: str = 'Gráfico'):
    """Plotagem de pontos e de uma função.

    Plotagem dos pontos indicados pelas coordenadas x e y, seguindo a
    função f. Função privada, auxiliar da função principal lin_interp.

    Args:
        x: lista das coordenadas x, em x[i], de cada ponto i.
        y: lista das coordenadas y, em y[i], de cada ponto i.
        f: função que será plotada.

    Returns:
        None
    """
    x_points = np.linspace(x[0], x[-1], 500)
    y_points = [f(xp) for xp in x_points]

    _, ax = plt.subplots()
    ax.scatter(x, y, color = 'red', label = 'Dados')
    ax.plot(x_points, y_points, label = 'Interpolação Linear por Partes')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(titulo)
    ax.grid(True)
    plt.show()

    return

def vandermond_interp(x: list,
                      y: list,
                      plot: bool = False) -> Callable:
    if len(x) != len(y):
            raise RuntimeError(f"x e y devem ter a mesma quantidade de elementos")
    
    x, y = _ordenar_coordenadas(x, y)
    
    n = len(x)
    matrix_vandermond = np.ones([n, n])
    for i in range(n):
        for k in range(1, n):
            matrix_vandermond[i][k] = matrix_vandermond[i][k-1] * x[i]
    
    coef = np.linalg.solve(matrix_vandermond, y)
    
    def f(x1: float) -> float:
        y1 = 0
        for a in coef[::-1]:
            y1 = a + y1*x1

        return y1

    # Plotagem do gráfico correspondente à função f
    if plot:
        _plotar(x, y, f, 'Interpolação Linear por Partes')

    return f