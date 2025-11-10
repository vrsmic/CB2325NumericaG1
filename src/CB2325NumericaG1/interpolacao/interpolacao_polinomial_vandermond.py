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

    Raises:
        RuntimeError: caso x e y não tenham mesma quantidade de elementos.
    """
    if len(x) != len(y):
        raise RuntimeError(f"x e y devem ter a mesma quantidade de elementos")
    
    x_np = np.array(x)
    y_np = np.array(y)

    idx = np.argsort(x_np)

    x_ord = x_np[idx]
    y_ord = y_np[idx]

    return x_ord, y_ord

def _random_sample(intv: list, N: int) -> np.array:
    """Cria uma amostra aleatória.
    
    Args:
        intv: intervalo da amostra.
        N: número de amostras.

    Returns:
        Lista numpy.array das amostras.
    """
    r = np.random.uniform(intv[0], intv[1], N-2)
    r.sort()
    return np.array([intv[0]] + list(r) + [intv[1]])

def _error_pol(f: Callable,
               P: Callable,
               intv: list,
               n: int = 1000) -> np.array:
    """Calcula o erro médio e o erro máximo
    
    Args:
        f: função analisada.
        P: função idealizada.
        intv: intervalo analisado.
        n = quantidade de amostras.

    Reuturns:
        Erro médio.
        Erro máximo.
    """
    x = _random_sample(intv, n)
    error = np.abs(f(x)-P(x))
    return np.sum(error)/n, np.max(error)

def _plotar(x: list,
            y: list,
            f: Callable,
            titulo: str = 'Gráfico',
            f_ideal: Callable = None):
    """Plotagem de pontos e de uma função.

    Plotagem dos pontos indicados pelas coordenadas x e y, seguindo a
    função f. Caso haja uma função ideal, ela também é plotada, e seu
    erro é calculado. Função privada, auxiliar da função principal lin_interp.

    Args:
        x: lista das coordenadas x, em x[i], de cada ponto i.
        y: lista das coordenadas y, em y[i], de cada ponto i.
        f: função que será plotada.
        titulo: titulo do gráfico que será plotado.
        f_ideal: função ideal (caso tenha).

    Returns:
        None
    """
    x_points = np.linspace(x[0], x[-1], 500)
    y_points = [f(xp) for xp in x_points]

    _, ax = plt.subplots()
    ax.scatter(x, y, color = 'red', label = 'Dados')
    ax.plot(x_points, y_points, 'b-', linewidth=2, label = 'Interpolação Polinomial (Vandermonde)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)

    if f_ideal: # Caso exista uma função ideal, o erro médio é calculado e mostrado na plotagem
        y_ideal = [f_ideal(xp) for xp in x_points]
        intv = [x[0], x[1]]
        erroMedio, erroMax = _error_pol(f, f_ideal, intv, n= 1000)
        ax.set_title(titulo+f' - Erro Médio = {erroMedio:2.4f} - Erro Máximo = {erroMax:2.4f}')
        ax.plot(x_points, y_ideal, color = 'g', label = 'Função ideal')
    else:
        ax.set_title(titulo)
    
    ax.legend()
    plt.show()

    return

def vandermond_interp(x: list,
                      y: list,
                      plot: bool = False,
                      f_ideal: Callable = None) -> Callable:
    """Interpolação polinomial pelo método de Vandermonde

    Essa função ordena pontos a partir da ordem crescente das
    coordenadas x. Em seguida, cria matriz de Vandermonde e retorna
    a solução algébrica para, assim, conseguir os coeficientes do
    polinômio interpolado. Caso haja uma função ideal, também é calculado
    o erro, que é representado no plot. Por fim, caso 'plot = True', há uma
    plotagem do gráfico correspondente. Por padrão, 'plot = False',
    ou seja, por padrão não há a plotagem.

    Args:
        x: lista das coordenadas x, em x[i], de cada ponto i.
        y: lista das coordenadas y, em y[i], de cada ponto i.
        plot: indica se deve haver a plotagem (True) ou não (False).
        f_ideal: função ideal, caso queira fazer comparação de erros.

    Returns:
        f: função de interpolação linear por partes.

    Raises:
        RuntimeError: caso x e y não tenham mesma quantidade de elementos.
    """

    # Verifica se x e y têm a mesma quantidade de elementos
    if len(x) != len(y):
            raise RuntimeError(f"x e y devem ter a mesma quantidade de elementos")
    
    # Ordenação das coordenadas x em ordem crescente
    x, y = _ordenar_coordenadas(x, y)
    
    n = len(x)
    matrix_vandermond = np.ones([n, n]) # Criação da matriz de Vandermonde com uns
    for i in range(n): # Eleva cada elemento da matriz ao seu devido expoente
        for k in range(1, n):
            matrix_vandermond[i][k] = matrix_vandermond[i][k-1] * x[i]
    
    coef = np.linalg.solve(matrix_vandermond, y)
    
    # Definição da função de interpolação
    def f(x1: float) -> float:
        y1 = 0
        for a in coef[::-1]: # Método de Horner para calcular valores de polinômios
            y1 = a + y1*x1

        return y1

    # Plotagem do gráfico correspondente à função f
    if plot:
        if f_ideal: # Caso exista uma função ideal, o erro é "plotado" no título
            _plotar(x, y, f, 'Interpolação Polinomial (Vandermonde)', f_ideal= f_ideal)
        else:
            _plotar(x, y, f, 'Interpolação Polinomial (Vandermonde)')

    return f