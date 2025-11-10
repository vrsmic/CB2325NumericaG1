import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
        
def bissecao(function: Callable, lower: float, upper: float, tolerance: float, plot: bool = True) -> float:
    '''
    Calcula a raiz aproximada de uma função usando o método da bisseção.

    =============

    Parâmetros:

    function : Função cuja raiz queremos encontrar, deve aceitar apenas um argumento x.
    lower : Limite inferior do intervalo em que queremos calcular a raiz da função.
    upper : Limite superior do intervalo em que queremos calcular a raiz da função.
    tolerance : Valor mínimo que o intervalo pode assumir.

    ============

    Retorna:
    Valor aproximado da raiz da função, arredondado para 4 casas decimais.
    '''
    
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
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()

    final_root = (lower + upper) / 2
    return round(final_root, 4)