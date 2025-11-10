import numpy as np
import matplotlib.pyplot as plt
from typing import Callable



def poly_interp(x_val: list,
                y_val: list,
                plot: bool = False,
                res: int = 100):
    """
    Interpolação polinomial a partir do método de Lagrange.

    Parâmetros:
        x_val: lista das coordenadas x.
        y_val: lista das coordenadas y.
        plot: indica se deve haver a plotagem (True) ou não (False).
        res: número de pontos para a plotagem do polinômio.

    Retorna:
        Uma função P que corresponde ao polinômio interpolador.
    """

    n = len(x_val)

    x_val_np = np.array(x_val)
    y_val_np = np.array(y_val)

    def P(x: int):
        acc_som = 0

        # loop do somatório
        for j in range(n):          
            # o produtorio exije que j != i
            x_val_sj = np.delete(x_val_np, j)

            # calculando numeradores e denominadores do produtorio
            num = x - x_val_sj
            den = x_val_np[j] - x_val_sj

            acc_som += y_val_np[j] * np.prod(num / den)
        
        return acc_som
    
    # plotagem do grafico caso o usuario deseje
    if plot:
        _poly_interp_plotter(x_val, y_val, P, res)

    return P


def _poly_interp_plotter(x_val: list, y_val: list, P: Callable, res=100):
    """
    Plotar gráfico da interpolação polinomial.

    Parâmetros:
        x_val: lista das coordenadas x.
        y_val: lista das coordenadas y.
        P: função que corresponde ao polinômio interpolador.
        res: número de pontos para a plotagem do polinômio.
    """
    # pontos recebidos
    plt.scatter(x_val, y_val, color="red", label='Pontos Originais', zorder=5)

    # gerando e plotando os pontos do polinômio
    x_plot = np.linspace(min(x_val), max(x_val), res)
    y_plot = [P(x) for x in x_plot]
    plt.plot(x_plot, y_plot)
    plt.title("Interpolação Polinomial de Lagrange")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

# testes

pol = poly_interp([1, 3, 5, 5, 6], [2, 4, 6, 8, 10], plot=True)

print(pol(87))


# tenho que comentar e melhorar o nome das variaveis; codigo precisa estar legivel

# para o matplot, preciso fazer algo que gere vários pontos do polinomio dado, e que o numero de pontos seja alteravel