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
    Raises:
    """
    
    x_points = np.linspace(x[0], x[-1], 500)
    y_points = [f(xp) for xp in x_points]

    _, ax = plt.subplots()
    ax.scatter(x, y, color = 'red', label = 'Dados')
    ax.plot(x_points, y_points,'b-', linewidth=2, label = 'Interpolação Linear por Partes')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(titulo)
    ax.grid(True)
    plt.show()

    return

def lin_interp(x: list,
               y: list,
               plot: bool = False
               ) -> Callable:
    """Interpolação linear por partes a partir dos pontos dados.

    Essa função ordena pontos a partir da ordem crescente das
    coordenadas x. Em seguida, cria retas descritas pela nova
    função f, que 'ligam' os pontos descritos pelas coordenadas x e y.
    É permitida extrapolação. Por fim, caso 'plot = True', há uma
    plotagem do gráfico correspondente.

    Args:
        x: lista das coordenadas x, em x[i], de cada ponto i.
        y: lista das coordenadas y, em y[i], de cada ponto i.
        plot: indica se deve haver a plotagem (True) ou não (False).

    Returns:
        f: função de interpolação linear por partes
    Raises:
        ValueError: Caso 'x' e 'y' tenham tamanhos diferentes, caso as listas
        estejam vazias, ou caso as coordenadas em 'x' não sejam distintas.
        TypeError: caso 'x' ou 'y' não sejam listas, ou caso 'plot não seja bool'.
    """
    # Tratamento de erros
    try:
        n = len(x)
        m = len(y)
    except:
        raise TypeError("Os argumentos 'x' e 'y' devem ser listas.")
    if type(plot) != bool:
        raise TypeError("O argumento 'plot' deve ser bool")
    
    if n != m:
        raise ValueError("As listas de coordenadas x e y devem ter o mesmo tamanho.")
    if n == 0:
        raise ValueError("As listas x e y não podem estar vazias.")
    if len(set(x)) != n:
        raise ValueError("As coordenadas x devem ser todas distintas.")

    # Ordenação das coordenadas x em ordem crescente
    x, y = _ordenar_coordenadas(x, y)

    # Definição da função de interpolação
    def f(x1: float) -> float:
        if x1 < x[0]: # Caso fora do intervalo, continua a reta mais próxima
            a = (y[1] - y[0])/(x[1] - x[0])
            b = y[0]
                    
            y1 = b + (x1 - x[0]) * a 
            return y1
        elif x1 > x[-1]: # Caso fora do intervalo, continua a reta mais próxima
            a = (y[-1] - y[-2])/(x[-1] - x[-2])
            b = y[-2]
                    
            y1 = b + (x1 - x[-2]) * a
            return y1
        else:
            for i in range (1, len(x)): # Busca intervalo do número x1 para atribuir valor f(x1)
                if x[i] >= x1 >= x[i-1]:
                    a = (y[i] - y[i-1])/(x[i] - x[i-1])
                    b = y[i-1]
                    
                    y1 = b + (x1 - x[i-1]) * a # Aproximação linear
                    
                    return y1

    # Plotagem do gráfico correspondente à função f
    if plot:
        _plotar(x, y, f, 'Interpolação Linear por Partes')

    return f