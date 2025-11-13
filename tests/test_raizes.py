# Biblioteca padrão
import math

# Bibliotecas externas
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pytest
from typing import Callable, Union
from pytest import approx

# Funções a serem testadas
from CB2325NumericaG1.raizes import bissecao, newton_raphson, secante

# ====== TESTES DA BISSEÇÃO ======

def test_bissecao_funcao_seno():
    f = lambda x: np.sin(x)
    limite_inferior = 1
    limite_superior = 6
    tolerancia = 1e-7
    assert bissecao(f, limite_inferior, limite_superior, tolerancia, plot=False) == 3.1416

def test_bissecao_exponencial():
    f = lambda x: np.exp(x) - 2
    limite_inferior = 0
    limite_superior = 1
    tolerancia = 1e-8
    assert bissecao(f, limite_inferior, limite_superior, tolerancia) == 0.6931

def test_bissecao_funcao_impropria_para_o_metodo():
    f = lambda x: x**2
    limite_inferior = -2
    limite_superior = 2
    tolerancia = 1e-8
    
    with pytest.raises(ValueError, match="A função não tem sinais opostos nos limites do intervalo."):
        bissecao(f,limite_inferior,limite_superior,tolerancia)

def test_bissecao_tolerancia_menor_que_zero():
    f = lambda x: np.sin(x)
    limite_inferior = 1
    limite_superior = 6
    tolerancia = 0

    with pytest.raises(ValueError,match="Valor de tolerância inválido."):
        bissecao(f, limite_inferior, limite_superior,tolerancia)

# ====== TESTES DE NEWTON-RAPHSON ======
x = sp.symbols('x')
y = sp.symbols('y')

def test_newton_raphson_funcao_com_mais_de_uma_variavel():
    f = sp.sympify("sin(x) + cos(y)")
    chute = 2
    tolerancia = 1e-12

    with pytest.raises(ValueError,match= r"A expressão SymPy deve ter exatamente uma variável, mas foram encontradas 2: .*"):
        newton_raphson(f,chute,tolerancia)

def test_newton_raphson_funcao_de_tipo_incorreto():
    f = 'sin(x) + cos(y)'
    chute = 2
    tolerancia = 1e-12

    with pytest.raises(TypeError, match="function deve ser Callable ou sp.Basic."):
        newton_raphson(f,chute,tolerancia)

def test_newton_raphson_value_error_nan():
  # resultará em np.log(-5) -> NaN
  f = lambda x: np.log(x)-10
  chute = -5.0
  tolerancia = 1e-8

  with pytest.raises(ValueError):
      newton_raphson(f, chute, tolerancia)

def test_newton_raphson_value_error_pos_inf():
   # resultará em 1.0/0.0 -> inf
   f = lambda x: np.divide(1.0, x)
   chute = 0.0
   tolerancia = 1e-8

   with pytest.raises(ValueError):
      newton_raphson(f,chute,tolerancia)

def test_newton_raphson_value_error_neg_inf():
  # resultará em np.log(0) -> -inf
  f = lambda x: sp.log(x)
  chute = 0.0
  tolerancia = 1e-8

  with pytest.raises(ValueError):
      newton_raphson(f,chute,tolerancia)


def test_newton_value_error_em_derivada_inf():
    # Usando 2*sqrt(x) para simplificar a derivada para 1/sqrt(x)
    funcao = 2 * sp.sqrt(x) 
    chute = 0.0
    tolerancia = 1e-8
    with pytest.raises(ValueError):
        newton_raphson(funcao, chute, tolerancia)

def test_newton_zero_division_error_em_derivada_zero():

    #Usando f(x) = cos(x), cuja derivada f'(x) = -sin(x) é 0 em x=0.
    funcao = sp.cos(x)
    chute = 0.0
    tolerancia = 1e-8
    with pytest.raises(ZeroDivisionError):
        newton_raphson(funcao, chute, tolerancia)

def test_newton_raises_runtime_error_on_no_convergence():

    #Usando f(x) = x^3 - 2x + 2 com chute x0=0, que oscila entre 0 e 1.
    funcao = x**3 - 2*x + 2
    chute = 0.0
    tolerancia = 1e-8
    with pytest.raises(RuntimeError):
        newton_raphson(funcao, chute, tolerancia)

def test_newton_raphson_funcao_sem_problemas():

    f = sp.sympify("x**4 - 4*x**2 + 4")
    chute = 1.5
    tolerancia = 1e-12
    assert newton_raphson(f,chute,tolerancia) == 1.4142139095356205

# ====== TESTES DA SECANTE ======
def test_secante_type_error():
    with pytest.raises(TypeError):
        secante("não é função", 1.0, 2.0, 1e-6)

def test_secante_value_error_nan_inicial():
    def f_nan(x):
        if x == 1.0:
            return np.nan
        return x**2 - 2
    with pytest.raises(ValueError):
        secante(f_nan, 1.0, 2.0, 1e-6)

def test_secante_value_error_inf_mid_iteration():
    def f_inf(x):
        # Vai causar Inf na segunda iteração
        return 1.0 / (x - 1.0)
    with pytest.raises(ValueError):
        secante(f_inf, 0.0, 2.0, 1e-6)

def test_secante_zero_division_error():
    def f_flat(x):
        # Função com valores iguais para provocar diferença ~0
        return 1.0
    with pytest.raises(ZeroDivisionError):
        secante(f_flat, 1.0, 2.0, 1e-6)

def test_secante_runtime_error():
    def f_no_root(x):
        # Função que não cruza o eixo x
        return np.exp(x) + 10
    with pytest.raises(RuntimeError):
        secante(f_no_root, 0.0, 1.0, 1e-12)

# Teste sem erros:
def test_secante_successo():
    def f(x):
        return x**2 - 2
    root = secante(f, 1.0, 2.0, 1e-6)
    assert abs(root - np.sqrt(2)) < 1e-6
