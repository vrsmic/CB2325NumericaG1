import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math as mt
from typing import Callable


def monte_carlo(f: Callable[[float], float],inicio: float,final: float,n: int,plot: bool = False) -> float:
    """
    Calcula a integral aproximada de uma função univariada utilizando o método de Monte Carlo.

    Este método estima a área sob a curva de uma função f(x) no intervalo [inicio, final]
    por meio da geração de amostras aleatórias e do cálculo do valor médio da função.

    Args:
        f (Callable[[float], float]):
            Função a ser integrada. Deve aceitar apenas um argumento escalar 'x'.
        inicio (float):
            Limite inferior da integral.
        final (float):
            Limite superior da integral.
        n (int):
            Número de pontos aleatórios (amostras) utilizados na aproximação.
        plot (bool, optional):
            Se True, exibe o gráfico da função e da área equivalente à integral.
            Padrão é False.

    Returns:
        float:
            Valor aproximado da integral, arredondado para 4 casas decimais.

    Roots:
        ValueError:
            Se 'n' for menor ou igual a zero.
        TypeError:
            Se 'f' não for uma função chamável.


    Notes:
        - A função gera 'n' amostras uniformemente distribuídas no intervalo [inicio, final].
        - O resultado é uma estimativa estocástica da integral — portanto,
          valores diferentes de 'n' podem produzir pequenas variações nos resultados.
        - Se o parâmetro 'plot' for ativado, o gráfico exibirá:
            * A função f(x) em preto;
            * A média dos valores de f(x) (linha azul);
            * A área equivalente à integral em azul-claro;
            * A área sob a curva em verde.
    """

    if n <= 0:
        raise ValueError("O número de pontos 'n' deve ser maior do que 0.")

    if not callable(f):
        raise TypeError("O argumento 'f' deve ser uma função chamável.")

    # Soma acumulada dos valores da função em pontos aleatórios
    soma_valores = 0.0

    for _ in range(n):
        x_rand = random.uniform(inicio, final)
        soma_valores += f(x_rand)

    # Valor médio da função e cálculo da área estimada
    media_f = soma_valores / n
    area = abs(final - inicio) * media_f

    if plot:
        xs = np.linspace(inicio, final, 400)
        ys = np.array([f(x) for x in xs])
        ys_media = np.full_like(xs, media_f)

        # Criação de dois subplots lado a lado
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # --- Gráfico 1: Função original ---
        axs[0].plot(xs, ys, 'r-', label='f(x)')
        axs[0].axhline(0, color='black', linewidth=1)
        axs[0].set_title("Função original f(x)")
        axs[0].fill_between(xs, ys, color='green', alpha=0.4, label='Área sob a curva')
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("f(x)")
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # --- Gráfico 2: Áreas equivalentes ---
        axs[1].plot(xs, ys, 'k-', label='f(x)')
        axs[1].plot(xs, ys_media, 'b--', label='Média de f(x)')
        axs[1].fill_between(xs, ys_media, color='blue', alpha=0.5, label='Área equivalente')
        axs[1].set_title(f"Aproximação de Monte Carlo\nÁrea ≈ {area:.4f}")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("f(x)")
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return round(area, 4)


def monte_carlo_two_variables(f: Callable[[float, float], float],inicio_x: float,final_x: float,inicio_y: float,final_y: float,n: int,plot: bool = False) -> float:
    """
    Calcula a integral dupla aproximada de uma função de duas variáveis
    utilizando o método de Monte Carlo.

    A integral é estimada por amostragem aleatória uniforme sobre o retângulo
    definido por [inicio_x, final_x] * [inicio_y, final_y].

    Args:
        f (Callable[[float, float], float]):
            Função a ser integrada. Deve aceitar dois argumentos 'x' e 'y'.
        inicio_x (float):
            Limite inferior no eixo x.
        final_x (float):
            Limite superior no eixo x.
        inicio_y (float):
            Limite inferior no eixo y.
        final_y (float):
            Limite superior no eixo y.
        n (int):
            Número de pontos aleatórios (amostras) utilizados na aproximação.
        plot(bool):
            Se 'True', exibe um gráfico 3D da superfície f(x, y)
            e dos pontos amostrados. Padrão é 'False'.

    Returns:
        float:
            Valor aproximado da integral dupla (volume sob a superfície),
            arredondado para 4 casas decimais.

    Roots:
        ValueError:
            Se 'n' for menor ou igual a zero.
        TypeError:
            Se 'f' não for uma função chamável.
    """

    if n <= 0:
        raise ValueError("O número de pontos 'n' deve ser maior do que 0.")

    if not callable(f):
        raise TypeError("O argumento 'f' deve ser uma função chamável.")

    soma_valores = 0.0
    
    #Vetores para plotagem 3D mais tarde.
    pontos_x = []
    pontos_y = []
    pontos_z = []

    # Determinação dos pontos para o cálculo da área.
    for _ in range(n):
        x_rand = random.uniform(inicio_x, final_x)
        y_rand = random.uniform(inicio_y, final_y)
        soma_valores += f(x_rand, y_rand)
        pontos_x.append(x_rand)
        pontos_y.append(y_rand)
        pontos_z.append(f(x_rand,y_rand))

    media_f = soma_valores / n
    area_dominio = abs(final_x - inicio_x) * abs(final_y - inicio_y)
    volume = media_f * area_dominio
    
    if plot:
        # Cria uma malha regular para desenhar a superfície
        X = np.linspace(inicio_x, final_x, 50)
        Y = np.linspace(inicio_y, final_y, 50)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Superfície suave
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')

        # Pontos amostrados
        ax.scatter(pontos_x, pontos_y, pontos_z, color='red', s=15, label='Amostras')

        # Base do domínio
        ax.plot([inicio_x, final_x, final_x, inicio_x, inicio_x],
                [inicio_y, inicio_y, final_y, final_y, inicio_y],
                [0, 0, 0, 0, 0],
                color='black', linewidth=1)

        ax.set_title(f"Método de Monte Carlo 2D\nVolume aproximado ≈ {volume:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    return round(volume, 4)


