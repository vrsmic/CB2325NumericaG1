
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def trapezio(
    f : Callable[[float], float],
    inicio : float, 
    final : float, 
    n : int, 
    plot : bool = False
    )-> float :

    """
    Calcula a integral aproximada de uma função usando o método trapezoidal.

    Caso desejado, a função também plota os trapézios utilizados na aproximação
    e o gráfico da função original.

    Args:
        f (Callable[[float], float]): 
            Função a ser integrada. Deve aceitar apenas um argumento escalar `x`.
        inicio (float): 
            Limite inferior da integral.
        final (float): 
            Limite superior da integral.
        n (int): 
            Número de subintervalos (trapézios) utilizados na aproximação.
        plot (bool, optional): 
            Se `True`, exibe o gráfico da função e dos trapézios. 
            Padrão é `False`.

    Returns:
        float: 
            Valor aproximado da integral, arredondado para 4 casas decimais.

    Raises:
        ValueError: 
            Se `n` for menor ou igual a zero.
        TypeError:
            Se `f` não for uma função chamável.

    Dependencies:
        - `numpy` (importado como `np`)
        - `matplotlib.pyplot` (importado como `plt`)

    Notes:
        - O gráfico mostra a função original em vermelho e os trapézios da 
          aproximação em azul.
    """

    if n <= 0 :
        raise ValueError("O número de subintervalos 'n' deve ser maior do que 0.")
    
    if not callable(f) :
        raise TypeError("O argumento 'f' deve ser uma função chamável.")

    # Lista de pontos no intervalo [inicio, final].
    x = np.linspace(inicio, final, n+1)
    y = np.array([f(xi) for xi in x])

    # Passo entre pontos.
    step = (final - inicio) / n

    # Aplica a fórmula do trápezio.
    integral_total = step * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])

    if plot :

        # Plota os trapézios.
        for i in range(n) :
            xi = x[i]
            yi = y[i]
            xii = x[i+1]
            yii = y[i+1]
            xs = [xi, xi, xii, xii]
            ys = [0, yi, yii, 0]

            plt.fill(xs, ys, color='blue', edgecolor='black', alpha=0.7)


        # Plota a função original em vermelho.    
        plt.plot(x, y, color = 'red', linewidth = 1, label = 'f(x)') 

        # Plota o eixo x.
        plt.axhline(0, color='black', linewidth=1)

        # Configuração do gráfico
        plt.axis('equal')
        plt.title("Integração pelo método do trapézio")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

    return round(integral_total, 4)


def simpson13(
    f : Callable[[float], float] ,
    inicio : float,
    final : float, 
    n : int, 
    plot : bool = False
    ) -> float :

    """
    Calcula a integral aproximada de uma função utilizando o método de Simpson 1/3.

    Caso desejado, a função também plota o gráfico da aproximação com as parábolas
    utilizadas e a curva original da função.

    Args:
        f (Callable[[float], float]): 
            Função a ser integrada. Deve aceitar apenas um argumento escalar `x`.
        inicio (float): 
            Limite inferior da integral.
        final (float): 
            Limite superior da integral.
        n (int): 
            Número de subintervalos. Deve ser um número par.
        plot (bool, optional): 
            Se `True`, exibe o gráfico da função e das parábolas de aproximação. 
            Padrão é `False`.

    Returns:
        float: 
            Valor aproximado da integral, arredondado para 4 casas decimais.

    Raises:
        TypeError: 
            Se `f` não for uma função chamável.
        ValueError: 
            Se `n` não for par ou se for menor ou igual a zero.

    Dependencies:
        - `numpy` (importado como `np`)
        - `matplotlib.pyplot` (importado como `plt`)

    Notes:
        - O gráfico mostra a função original em vermelho e as parábolas 
          de aproximação em azul.
    """

    if not callable(f) :
        raise TypeError("O argumento 'f' deve ser uma função chamável.")

    if n % 2 != 0 or n <= 0 :
        raise ValueError("O número de intervalos n deve um par maior do que 0.")
    
    x = np.linspace(inicio, final, n+1)
    y = np.array([f(xi) for xi in x])

    # Passo entre pontos.
    step = (final - inicio) / n

    # Aplica a fórmula do parábolas.
    integral_total = y[0] + y[-1] + 4 * sum(y[1 : -1: 2]) + 2 * sum(y[2: -2: 2])
    integral_total *= step/3

    if plot :
        # Plota as parábolas.
        for i in range(0, n, 2) :
            xi = x[i:i+3]
            yi = y[i:i+3]
            coef = np.polyfit(xi, yi, 2)
            xs = np.linspace(xi[0], xi[-1], 50)
            ys = np.polyval(coef, xs)
            plt.fill_between(xs, ys, color='blue', alpha=0.7)


        # Plota a função original em vermelho.    
        plt.scatter(x, y, color = 'red', label = 'f(x)', s = 5)

        # Plota o eixo x.
        plt.axhline(0, color='black', linewidth=1)

        # Configuração do gráfico
        plt.axis('equal')
        plt.title("Integração pelo método do Simpson 1/3")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

    return round(integral_total, 4)