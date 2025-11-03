import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
        
def bissecao(function: Callable, lower: float, upper: float, tolerance: float) -> float:

    '''
    Calcula a raiz aproximada de uma função usando o método da bisseção.

    =============

    Parâmetros :

    function : Função cuja raiz queremos encontrar, deve aceitar apenas um argumento x.

    lower : Limite inferior do intervalo em que queremos calcular a raiz da funçãp.

    upper : Limite superior do intervalo em que queremos calcular a raiz da função.

    tolerance : Valor mínimo que o intervalo pode assumir

    ============

    Retorna : 
    - Valor aproximado da raiz da função, arredondado para 3 casas decimais

    '''
    
    # Listas para salvar os dados para o gráfico. 
    lower_record = []
    upper_record = []
    # Adiciona nelas os valores iniciais
    lower_record.append(lower)
    upper_record.append(upper)

    # Avalia a função nos pontos limites do intervalo
    lower_bound = function(lower)
    upper_bound = function(upper)
        
    # Verifica se um dos limites do intervalo é raiz
    if lower_bound == 0:
        return round(lower_bound, 4)
    elif upper_bound == 0:
        return round(upper_bound, 4)
            
    # Verifica se a função cumpre as condições para a utilização desse método
    if lower_bound * upper_bound > 0:
        raise ValueError("A função não tem sinais opostos nos limites do intervalo.")
    
    while upper-lower > tolerance:
        
        #Calcula o ponto médio do intervalo
        medium_point = (lower + upper) / 2
        medium_value = function(medium_point)
        
        if medium_value == 0:
            return round(medium_point, 4)
        
        elif  lower_bound * function(medium_point) < 0:

            upper = medium_point
        else:
            lower = medium_point
            
        # Adiciona os novos pontos às listas de histórico
        lower_record.append(lower)
        upper_record.append(upper)
    
    x = np.linspace(lower_record[0], upper_record[0], 100)
    y = function(x)

    # Plota o eixo x.
    plt.axhline(0, color='dimgrey', linewidth=1.5)
    
    # Plota a função original em preto.
    plt.plot(x, y, color = 'black', linewidth = 1.0, label = 'f(x)') 

    lower_points_y = function(np.array(lower_record))
    upper_points_y = function(np.array(upper_record))

    plt.scatter(lower_record, lower_points_y, s=20.0, c='royalblue', label='Limite Inferior', zorder=2)
    plt.xlim(lower_record[0], upper_record[0])
    plt.scatter(upper_record, upper_points_y,s=20.0, c='crimson', label='Limite Superior', zorder=2)


    # Configuração do gráfico
    # plt.axis('equal')
    plt.title("Raízes da função pelo Método da Bisseção")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()

    # Quando o loop para, a função retorna a melhor estimativa, isto é, o ponto médio do último intervalo
    final_root = (lower + upper) / 2
    return round(final_root, 4)

# teste de commit
