import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt

def regressao_linear(x, y):

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

  return a, b

def taylor(function, x_symbol, point, times):
    """
        Calcula e imprime o Polinômio de Taylor de uma função simbólica.

        Esta função gera o polinômio que aproxima a função 'function' em torno de um ponto 'point'.

        Args:
            function: A função simbólica a ser aproximada.
                Ex: sp.exp(x), sp.sin(x), sp.E**x
            x_symbol: A variável simbólica principal da função.
                Ex: sp.symbols('x')
            point (float ou int): O ponto 'a' (centro) em torno do qual a série será expandida.
            times (int): O número de termos do polinômio. A ordem máxima (grau)
                do polinômio resultante será de 'times - 1'

        Returns:
            None. A função imprime o polinômio simbólico resultante no console.
        """
    
    f_poly = 0 # Inicia o polinômio, teremos que adicionar os termos aqui depois.

    for i in range(times):
        # Calcula a i-ésima derivada simbólica (f"'(x))
        derivada_i_simbolica = sp.diff(function, x_symbol, i) # Calcula a i-ésima derivada.
        derivada_no_ponto = derivada_i_simbolica.subs(x_symbol, point) # Avalia a derivada.
        fatorial = math.factorial(i) # Calcula o fatorial.

    termo_polinomio = (derivada_no_ponto * (x_symbol - point)**i) / fatorial
    f_poly += termo_polinomio

    print(f"O Polinômio de Taylor com {times} termos é:")
    print(f_poly)