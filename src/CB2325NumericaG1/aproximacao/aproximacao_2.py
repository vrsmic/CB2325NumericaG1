import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def ajuste_linear(x: List[float], y: List[float], plot: bool = False) -> Tuple[float, float]:
    '''Calcula o ajuste linear (regressão linear) de um conjunto de pontos
    (x, y) usando o método dos mínimos quadrados encontra os coeficientes 'a'
    e 'b' para a reta y = ax + b.

    Parâmetros :
        x : Lista ou array de valores para o eixo x.
        y : Lista ou array de valores para o eixo y.
    
    plot : Se True, plota o gráfico com os pontos
           originais e a reta ajustada. Padrão é False.

    Retorna :
          Uma tupla (a, b) contendo os coeficientes da reta
          (coeficiente angular 'a' e intercepto 'b').
    
    Observações :
        x e y devem ter o mesmo número de elementos.
    '''

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
        x_line = np.linspace(np.min(x_arr), np.max(x_arr), 100)
        y_line = a * x_line + b

        plt.figure(figsize=(10, 6))
        plt.scatter(x_arr, y_arr, color='red', label='Pontos Reais (Dados)')
        plt.plot(x_line, y_line, color='blue', linewidth=2,
                 label=f'Ajuste Linear: y = {a:.2f}x + {b:.2f}')
        plt.title("Ajuste Linear por Mínimos Quadrados")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.legend()
        plt.grid(True) 
        plt.show()

    return (a, b)

def ajuste_trigonometrico(x: List[float], y: List[float], periodo: float, plot: bool = False) -> Tuple[float, float, float]:
    '''
    Calcula o ajuste trigonométrico (Série de Fourier de 1ª ordem)
    de um conjunto de pontos (x, y) usando o método dos mínimos quadrados.
    Encontra os coeficientes c0, c1, c2 para a função:
    y = c0 + c1*cos(omega*x) + c2*sin(omega*x)
    onde omega = 2*pi / periodo.

    Parâmetros :
        x : Lista ou array de valores para o eixo x.
        y : Lista ou array de valores para o eixo y.
        periodo (T) : O período fundamental estimado dos dados. Estimar
                      corretamente é crucial para um bom ajuste.
        plot : Se True, plota o gráfico.

    Retorna :
        Uma tupla (c0, c1, c2) contendo os coeficientes do ajuste.
        (c0 = offset, c1 = amplitude do cosseno, c2 = amplitude do seno)
    '''
    x_arr = np.array(x)
    y_arr = np.array(y)

    if len(x_arr) != len(y_arr):
        raise ValueError("Os vetores x e y devem ter o mesmo tamanho.")
    if len(x_arr) < 3:
         raise ValueError("São necessários pelo menos 3 pontos para este ajuste (3 coeficientes).")
    if periodo <= 0:
        raise ValueError("O período deve ser um valor positivo.")

    # Calcular frequência angular
    omega = (2 * np.pi) / periodo

    # 1. Montar a matriz do sistema (Design Matrix) A
    # Queremos resolver Ac = y, onde c = [c0, c1, c2]
    # Cada linha de A é [1, cos(omega*xi), sin(omega*xi)]
    A = np.column_stack([
        np.ones_like(x_arr),    # Coluna para c0 (intercepto)
        np.cos(omega * x_arr),  # Coluna para c1 (cosseno)
        np.sin(omega * x_arr)   # Coluna para c2 (seno)
    ])

    # 2. Resolver o sistema de equações lineares por mínimos quadrados
    # c = (A^T A)^-1 * (A^T y)
    # O np.linalg.lstsq faz isso de forma eficiente e estável
    # Retorna (coeficientes, resíduos, rank, valores singulares)
    # Pegamos apenas o primeiro item (coeficientes) [0]
    try:
        coeficientes, _, _, _ = np.linalg.lstsq(A, y_arr, rcond=None)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Não foi possível resolver o sistema: {e}")

    c0, c1, c2 = coeficientes

    if plot:
        plt.figure(figsize=(10, 6))
        # Pontos originais
        plt.scatter(x_arr, y_arr, color='red', label='Pontos Reais (Dados)')

        # Gerar linha para o ajuste
        # Usamos mais pontos (500) para garantir uma curva suave
        x_line = np.linspace(np.min(x_arr), np.max(x_arr), 500) 
        y_line = c0 + c1 * np.cos(omega * x_line) + c2 * np.sin(omega * x_line)

        plt.plot(x_line, y_line, color='purple', linewidth=2,
                 label=f'Ajuste Trig: y = {c0:.2f} + {c1:.2f}cos(ωx) + {c2:.2f}sin(ωx)\n(Período T={periodo:.2f}, ω={omega:.2f})')
        
        plt.title("Ajuste Trigonométrico por Mínimos Quadrados")
        plt.xlabel("Eixo x")
        plt.ylabel("Eixo y")
        plt.legend()
        plt.grid(True)
        plt.show()

    return (c0, c1, c2)


if __name__ == "__main__":
    
    # --- Exemplo 1: Dados quase lineares ---
    print("--- Exemplo de Ajuste Linear ---")
    x_linear = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # y = 1.5x + 3 + ruído
    y_linear = [3.1, 4.4, 6.1, 7.4, 9.0, 10.6, 12.0, 13.4, 15.1, 16.3]

    # Chama a função de ajuste linear
    a, b = ajuste_linear(x_linear, y_linear, plot=True)
    print(f"Coeficientes do Ajuste Linear: a={a:.4f}, b={b:.4f}\n")


    # --- Exemplo 2: Dados trigonométricos ---
    print("--- Exemplo de Ajuste Trigonométrico ---")
    
    # Gerando dados senoidais com ruído
    # Vamos usar um período T = 10. Então omega = 2*pi/10
    T_real = 10.0
    omega_real = (2 * np.pi) / T_real
    
    x_trig = np.linspace(0, 20, 50) # 50 pontos de 0 a 20 (2 períodos)
    
    # y = 5.0 + 3.0*cos(wx) + 2.0*sin(wx) + ruído
    ruido = np.random.normal(0, 0.5, x_trig.shape) # Ruído gaussiano
    y_trig = 5.0 + 3.0 * np.cos(omega_real * x_trig) + 2.0 * np.sin(omega_real * x_trig) + ruido

    # Chama a função de ajuste trigonométrico
    # Note que precisamos *informar* o período que suspeitamos para os dados.
    # Se errarmos muito o período, o ajuste será ruim.
    c0, c1, c2 = ajuste_trigonometrico(x_trig, y_trig, periodo=T_real, plot=True)
    print(f"Coeficientes do Ajuste Trigonométrico:")
    print(f"c0 (offset) = {c0:.4f} (Esperado: 5.0)")
    print(f"c1 (cos)    = {c1:.4f} (Esperado: 3.0)")
    print(f"c2 (sin)    = {c2:.4f} (Esperado: 2.0)")