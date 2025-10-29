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