import numpy as np
import matplotlib.pyplot as pltwd
from typing import List, Tuple

def ajuste_linear(x: List[float], y: List[float], plot: bool = False) -> Tuple[float, float]:

    # Converter para arrays numpy
    x_arr = np.array(x)
    y_arr = np.array(y)

    if len(x_arr) != len(y_arr):
        raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")

    n = len(x_arr)
    if n == 0:
        raise ValueError("Os vetores x e y não podem estar vazios.")

    # Implementação do método dos mínimos quadrados
    # Usando as médias, que é numericamente mais estável

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    # Calcular 'a' (coeficiente angular)
    # a = Sxy / Sxx
    numerador = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominador = np.sum((x_arr - x_mean)**2)

    if denominador == 0:
        # Isso acontece se todos os valores de x forem iguais
        raise ValueError("Não é possível calcular a regressão: todos os valores de x são iguais.")

    a = numerador / denominador

    # Calcular 'b' (intercepto)
    # b = y_media - a * x_media
    b = y_mean - (a * x_mean)

    if plot:
        # Gerar pontos para a reta ajustada
        # Criamos uma linha que vai do x mínimo ao x máximo
        x_line = np.linspace(np.min(x_arr), np.max(x_arr), 100)
        y_line = a * x_line + b

        # Configurar o plot, inspirado no estilo do seu outro código
        plt.figure(figsize=(10, 6))

        # Pontos originais (dados)
        plt.scatter(x_arr, y_arr, color='red', label='Pontos Reais (Dados)')

        # Reta ajustada
        plt.plot(x_line, y_line, color='blue', linewidth=2,
                 label=f'Ajuste Linear: y = {a:.2f}x + {b:.2f}')

        # Configuração do gráfico
        plt.title("Ajuste de Função por Mínimos Quadrados")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.legend()
        plt.grid(True) # Adiciona um grid para facilitar a leitura
        plt.show()