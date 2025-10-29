import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

# Regressão Linear.
def regressao_linear(x, y, plot=False):
    """
    Calcula o coeficiente angular 'a' e o coeficiente linear 'b' de uma regressão linear simples (y = ax + b).

    A regressão é realizada utilizando o Método dos Mínimos Quadrados Ordinários (MQO).

    Parâmetros:
    ----------
    x : Vetor contendo os valores da variável independente.
    y : Vetor contendo os valores da variável dependente.

    Retorna:
    -------
    tuple
        Uma tupla contendo (a, b), onde:
        a (float): Coeficiente angular da linha de regressão.
        b (float): Coeficiente linear da linha de regressão.

    Levanta:
    -------
    ValueError: Se 'x' e 'y' tiverem comprimentos diferentes ou se tiverem menos de 2 pontos de dados, o que é insuficiente para a regressão.

    Dependências:
    ------------
    - É necessário ter a biblioteca `numpy` (importada como `np`).
    """
  # Verificando se os dados podem satisfazer as equações da regressão linear.
    if len(x) != len(y):
        raise ValueError("X e y devem ter o mesmo número de amostras.")
    if len(x) < 2:
        raise ValueError("A regressão requer pelo menos 2 pontos.")

    x_media = np.mean(x) # Calcula a média de x.
    y_media = np.mean(y) # Calcula a média de y.

    numerador = np.sum((x - x_media) * (y - y_media))
    denominador = np.sum((x - x_media)**2)
    a = numerador / denominador
    b = y_media - (a * x_media)

    if plot:
        x1 = [] # Nota: Também pode ser um np.array.
        y1 = [] # Nota: Também pode ser um np.array.

        print(f"Coeficientes encontrados:")
        print(f"Inclinação (a): {a:.3f}")
        print(f"Intercepto (b): {b:.3f}")

        x2 = np.linspace(min(x1), max(x1), 100) # cria 100 pontos entre o menor e o maior ponto de x1, para criar a reta do primeiro ponto ao fim.
        y2 = (a * x2) + b # Equação da reta de regressão.

        plt.scatter(x1, y1, color='red', label='Dados Originais') # Plot dos dados.
        plt.plot(x2, y2, color='blue', linewidth=2, label=f'Reta de Regressão: y = {a:.2f}x + {b:.2f}') # Plot da reta.

        x_margin = (max(x1) - min(x1)) * 0.1 # Calcula a margem de x.
        y_margin = (max(y1) - min(y1)) * 0.1 # Calcula a margem de y.

        # Plot
        plt.xlim(min(x1) - x_margin, max(x1) + x_margin) # Limita a margem de x.
        plt.ylim(min(y1) - y_margin, max(y1) + y_margin) # Limita a margem de y.
        plt.title("Regressão Linear")
        plt.xlabel("Eixo X")
        plt.ylabel("Eixo Y")
        plt.legend()
        plt.grid(False) # Grade quadriculada.
        plt.show()

    return a, b

# Polinômio de Taylor.
def polinomio_de_taylor(function, x_symbol, point, times, plot=False):
    """
    Calcula, imprime e opcionalmente plota o Polinômio de Taylor
    para uma dada função em torno de um ponto.

    Esta função utiliza a biblioteca SymPy para realizar a
    diferenciação simbólica e a construção do polinômio.
    Se plot = True, utiliza Numpy e Matplotlib para visualizar a aproximação.

    Parâmetros:
    ----------
    function: A expressão simbólica da função que será aproximada.

    x_symbol: O símbolO em relação ao qual o polinômio será construído e as derivadas calculadas.

    point (float ou int): O ponto de expansão 'a' (o centro) em torno do qual a função será aproximada.

    times (int): O número de termos a serem usados na série. O polinômio resultante terá grau 'times - 1'.

    plot (bool, opcional): Se True, gera um gráfico comparando a função original e o polinômio de Taylor. Por padrão é False.
    
    Retorna:
    -------
    sympy.Expr: A expressão simbólica do Polinômio de Taylor resultante.

    Plot (Opcional): Plot da função e da aproximação de Taylor.

    Dependências:
    ------------
    - É necessário ter as bibliotecas `sympy` (importada como `sp`)
      e `math` importadas no escopo.
      `numpy` (importado como `np`) - Necessário se plot=True
      `matplotlib.pyplot` (importado como `plt`) - Necessário se plot=True
    """

    f_poly = 0 # Inicia o polinômio, teremos que adicionar os termos aqui depois.

    for i in range(times):
            
            # Calcula a i-ésima derivada simbólica (f"'(x))
            derivada_i_simbolica = sp.diff(function, x_symbol, i)
            
            # Avalia a derivada no ponto 'a'
            derivada_no_ponto = derivada_i_simbolica.subs(x_symbol, point)
            
            # Calcula o fatorial
            fatorial = math.factorial(i)

            # Monta o i-ésimo termo do polinômio
            termo_polinomio = (derivada_no_ponto * (x_symbol - point)**i) / fatorial
            
            # Adiciona ao polinômio total
            f_poly += termo_polinomio

    print(f"O Polinômio de Taylor com {times} termos é:")
    print(f_poly)

def regressao_logaritmica(x, y, plot=False):
    """
    Calcula os coeficientes 'a' e 'b' de uma regressão logarítmica do tipo (y = a * ln(x) + b), e 
    optacionalmente plota a função de regressão

    Este método funciona transformando a variável preditora 'x' aplicando o logaritmo natural (ln),
    depois utilizando a função 'regressao_linear' padrão nos dados transformados (ln(x), y).

    Parâmetros:
    ----------
    x: Vetor contendo os valores da variável independente (preditora).
        IMPORTANTE: Todos os valores de 'x' devem ser estritamente
        positivos (x > 0), pois o logarítmo de valores negativos não é definido.
    
    y: Vetor contendo os valores da variável dependente (resposta).

    plot (bool, opcional): Se True, gera um gráfico comparando a função original e o polinômio de Taylor. Por padrão é False.

    Retorna:
    -------
    tuple
        Uma tupla contendo (a, b), onde:
        a (float): Coeficiente 'a' que multiplica o termo ln(x).
        b (float): Coeficiente 'b' (intercepto) da linha de regressão.
    
    Plot (Opcional): Plot dos pontos e da função de regressão.

    Levanta:
    -------
    ValueError
        Herdado da função 'regressao_linear':
        - Se 'x' e 'y' tiverem comprimentos diferentes.
        - Se tiverem menos de 2 pontos de dados.
    RuntimeWarning
        Pode ser levantado pelo `numpy` se 'x' contiver valores
        menores ou iguais a zero, resultando em 'NaN' ou '-inf'.
    """
    # Transforma a variável x aplicando o logaritmo natural
    log_x = np.log(x)

    # Usa a função de regressão linear já existente nos dados transformados.
    a, b = regressao_linear(log_x, y)

    return a, b