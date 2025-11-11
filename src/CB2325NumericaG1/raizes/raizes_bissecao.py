import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
        
def bissecao(function: Callable, lower: float, upper: float, tolerance: float, plot: bool = False) -> float:
    """
    Encontra/aproxima uma raiz de uma função real de variável real usando o método de Newton–Raphson.

    Args:
        function (Callable):
            Função cuja raíz queremos encontrar ou aproximar.
        lower (float):
            Limite inferior do intervalo em que queremos calcular a raiz da função.
        upper (float):
            Limite superior do intervalo em que queremos calcular a raiz da função.
        tolerance (float):
            Critério de parada.
            Valor mínimo que o intervalo pode assumir.
        plot (bool = False):
            Determina se uma visualização gráfica do método será plotada.
            Por padrão, não será.

    Returns:
        float:
            Valor aproximado da raiz.

    Raises:
        ValueError:
            Se a função não tem sinais opostos nos limites do intervalo.
            Se a tolerância não for positiva.
    """
    
    # Listas para salvar os dados para o gráfico
    lower_record = [lower]
    upper_record = [upper]

    # Avalia a função nos pontos limites do intervalo
    lower_bound = function(lower)
    upper_bound = function(upper)
        
    # Verifica se um dos limites do intervalo é raiz
    if lower_bound == 0:
        return round(lower, 4)
    elif upper_bound == 0:
        return round(upper, 4)
            
    # Verifica se a função cumpre as condições para a utilização desse método
    if lower_bound * upper_bound > 0:
        raise ValueError("A função não tem sinais opostos nos limites do intervalo.")
    if tolerance <= 0:
        raise ValueError("Valor de tolerância inválido.")
    
    while upper - lower > tolerance:
        medium_point = (lower + upper) / 2
        medium_value = function(medium_point)
        
        if medium_value == 0:
            return round(medium_point, 4)
        
        if lower_bound * medium_value < 0:
            upper = medium_point
            upper_bound = medium_value
        else:
            lower = medium_point
            lower_bound = medium_value
            
        lower_record.append(lower)
        upper_record.append(upper)
            
    if plot:
        x = np.linspace(lower_record[0], upper_record[0], 100)
        y = function(x)

        plt.axhline(0, color='dimgrey', linewidth=1.5)
        plt.plot(x, y, color='black', linewidth=1.0, label='f(x)') 

        plt.scatter(lower_record, function(np.array(lower_record)), s=20, c='royalblue', label='Limite Inferior', zorder=2)
        plt.scatter(upper_record, function(np.array(upper_record)), s=20, c='crimson', label='Limite Superior', zorder=2)

        plt.xlim(lower_record[0], upper_record[0])
        plt.title("Raízes da função pelo Método da Bisseção")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

    final_root = (lower + upper) / 2
    return round(final_root, 4)