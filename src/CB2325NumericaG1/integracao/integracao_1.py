
import numpy as np
import matplotlib.pyplot as plt

def trapezio(f, inicio, final, n) :

    '''
    Calcula a integral aproximada de uma função usando o método 
    trapezoidal e plota os trapézios usados na aproximação.

    ------------

    Parâmetros :

    f : Função a ser integrada, deve aceitar apenas um argumento x.

    inicio : Limite inferior da integral.

    final : Limite superios da integral.

    n : Número de subintervalos (trapézios).

    -----------

    Retorna : 
    - Valor aproximado da integral, arredondado
        para 4 casas decimais.
    
    -----------

    Observações : 
    - A função plota o gráfico com os trapézios da aproximação em azul.
    - Junto com o gráfico da função original eme vermelho.

    '''

    # Lista de pontos no intervalo [inicio, final].
    x = np.linspace(inicio, final, n+1)
    y = np.array([f(xi) for xi in x])

    # Passo entre pontos.
    step = (final - inicio) / n

    # Aplica a fórmula do trápezio.
    integral_total = step * (0.5*y[0] + sum(y[1:-1]) + 0.5*y[-1])

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


def simpson13(f, inicio, final, n) :
    '''
    Calcula a integral aproximada de uma função usando o método 
    de simpson 1/3 e plota as parábolas.

    ------------

    Parâmetros :

    f : Função a ser integrada, deve aceitar apenas um argumento x.

    inicio : Limite inferior da integral.

    final : Limite superios da integral.

    n : Número de subintervalos .

    -----------

    Retorna : 
    - Valor aproximado da integral, arredondado
        para 4 casas decimais.
    
    -----------

    Observações : 
    - A função plota o gráfico com as parábolas da aproximação em azul.
    - Junto com o gráfico da função.

    '''

    if n % 2 != 0 :
        raise ValueError("O número de intervalos n deve ser par.")
    
    x = np.linspace(inicio, final, n+1)
    y = np.array([f(xi) for xi in x])

    # Passo entre pontos.
    step = (final - inicio) / n

    # Aplica a fórmula do parábolas.
    integral_total = y[0] + y[-1] + 4 * sum(y[1 : -1: 2]) + 2 * sum(y[2: -2: 2])
    integral_total *= step/3


    # Plota as parábolas.
    for i in range(0, n, 2) :
        xi = x[i:i+3]
        yi = y[i:i+3]
        coef = np.polyfit(xi, yi, 2)
        xs = np.linspace(xi[0], xi[-1], 50)
        ys = np.polyval(coef, xs)
        plt.fill_between(xs, ys, color='blue', edgecolor='black', alpha=0.7)


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