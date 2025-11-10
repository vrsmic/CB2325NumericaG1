import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def _poly_interp_plotter(x_val: list,
                         y_val: list,
                         P: Callable,
                         res: int,
                         pcolor: str,
                         ccolor: str,
                         titulo: str) -> None:
    """
    Plotar gráfico da interpolação polinomial.

    Parameters
    ----------
        x_val : list
            lista das coordenadas x.
        y_val : list
            lista das coordenadas y.
        P : Callable
            função que corresponde ao polinômio interpolador.
        res : int
            número de pontos para a plotagem do polinômio.
        pcolor : str
            cor dos pontos originais.
        ccolor : str
            cor da curva (polinômio interpolador).
        titulo : str
            título do gráfico (string

    Returns
    -------
        None
    """

    # pontos recebidos
    plt.scatter(x_val, y_val, color=pcolor, label='Pontos Originais', zorder=5)

    # gerando e plotando os pontos do polinômio
    x_plot = np.linspace(min(x_val), max(x_val), res)
    y_plot = [P(x) for x in x_plot]
    plt.plot(x_plot, y_plot, color=ccolor, label='Interpolação Polinomial', linewidth=2)

    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

    return


def poly_interp(x_val: list,
                y_val: list,
                plot: bool = False,
                res: int = 100,
                pcolor: str = "#234883",
                ccolor: str = "#4287f5",
                titulo: str = "Interpolação Polinomial de Lagrange") -> Callable:
    """
    Gera um polinômio interpolador usando o método de Lagrange.

    Esta função recebe um conjunto de pontos (x, y) e retorna uma 
    função (polinômio) que passa exatamente por todos esses pontos.

    Parameters
    ----------
    x_val : list
        Uma lista de coordenadas x dos pontos. Os valores devem ser
        todos distintos entre si.
    y_val : list
        Uma lista de coordenadas y dos pontos. Deve ter o mesmo
        tamanho de `x_val`.
    plot : bool, optional
        Se True, exibe um gráfico do polinômio e dos pontos 
        originais. O padrão é False.
    res : int, optional
        Resolução (número de pontos) usada para desenhar a 
        curva do polinômio no gráfico. O padrão é 100.
    pcolor : str, optional
        Cor dos pontos originais no gráfico. O padrão é "#234883".
    ccolor : str, optional
        Cor da curva do polinômio no gráfico. O padrão é "#4287f5".
    titulo : str, optional
        Título do gráfico. O padrão é "Interpolação Polinomial de Lagrange".

    Returns
    -------
    Callable
        Uma função P(x) que recebe um número (int ou float) e 
        retorna o valor do polinômio interpolador avaliado 
        naquele ponto x.

    Raises
    ------
    ValueError
        - Se `x_val` e `y_val` tiverem tamanhos diferentes.
        - Se as listas de entrada estiverem vazias.
        - Se `x_val` contiver valores duplicados.
        - Se `res` não for um inteiro.
        - Se `pcolor` ou `ccolor` não forem strings.
        - Se `titulo` não for uma string.

    Examples
    --------
    >>> x = [0, 1, 2]
    >>> y = [1, 3, 2]
    >>> pol = poly_interp(x, y)
    >>> print(pol(1.5))
    2.625
    """

    # tratamento de erros
    n = len(x_val)
    m = len(y_val)

    if n != m:
        raise ValueError("As listas de coordenadas x e y devem ter o mesmo tamanho.")

    if n == 0:
        raise ValueError("Lista de pontos x inserida está vazia.")
    elif m == 0:
        raise ValueError("Lista de pontos y inserida está vazia.")
    
    if len(set(x_val)) != n:
        raise ValueError("As coordenadas x devem ser todas distintas.")
    
    if not (type(res) == int):
        raise ValueError("O argumento res deve ser um inteiro.")
    
    if not (type(pcolor) == str):
        raise ValueError("O argumento pcolor deve ser uma string.")
    
    if not (type(ccolor) == str):
        raise ValueError("O argumento ccolor deve ser uma string.")
    
    if not (type(titulo) == str):
        raise ValueError("O argumento titulo deve ser uma string.")
    

    
    # convertendo as arrays para o tipo do numpy
    x_val_np = np.array(x_val)
    y_val_np = np.array(y_val)


    def P(x: int | float) -> int | float:
        """
        Função que calcula o valor do polinômio interpolador em x.

        Parameters
        ----------
        x : int | float
            Ponto onde o polinômio será avaliado.

        Returns
        -------
        int | float
            O valor do polinômio interpolador em x.

        Raises
        ------
        ValueError
            Se x não for um número (int ou float).
        """

        if not np.isreal(x):
            raise ValueError("O argumento x deve ser um número real.")

        # acumula o termo y_j * L_j(x)
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
        _poly_interp_plotter(x_val, y_val, P, res, pcolor, ccolor, titulo)

    return P

