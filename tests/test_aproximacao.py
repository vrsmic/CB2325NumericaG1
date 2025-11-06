# Python 3

# Bibliotecas padrão
import math

# Bibliotecas de terceiros
import pytest
import numpy as np
import sympy as sp

# Funções a serem testadas (do outro arquivo)
from CB2325NumericaG1.aproximacao import regressao, regressao_linear, regressao_logaritmica, polinomio_de_taylor

TOLERANCIA = 1e-6 # Tolerância para comparações de float.

@pytest.fixture
def x_symbol():
    """Fixture para fornecer o símbolo 'x' do sympy."""
    return sp.Symbol('x')

def test_regressao_grau_2_ajuste_perfeito():
    """Testa uma regressão quadrática perfeita: y = x^2 - x + 2"""
    # y = 1*x^2 - 1*x + 2
    x = np.array([-1, 0, 1, 2, 3])
    y = np.array([4, 2, 2, 4, 8])
    grau = 2
    
    coefs = regressao(x, y, grau)
    
    esperado = np.array([2.0, -1.0, 1.0])
    
    assert coefs == pytest.approx(esperado, abs=TOLERANCIA)

def test_regressao_grau_1_equivale_a_linear():
    """Testa se a regressão de grau 1 produz o mesmo resultado da linear."""
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 3, 5, 7, 9]) # y = 2x + 1
    
    a_lin, b_lin = regressao_linear(x, y) # (a=2.0, b=1.0)
    
    # Teste com regressao grau 1
    coefs_reg = regressao(x, y, grau=1) # [c0, c1] -> [1.0, 2.0]
    
    assert coefs_reg[0] == pytest.approx(b_lin, abs=TOLERANCIA)
    assert coefs_reg[1] == pytest.approx(a_lin, abs=TOLERANCIA)

def test_regressao_error_comprimentos_diferentes():
    """Testa se levanta ValueError para comprimentos de x e y diferentes."""
    x = [1, 2, 3]
    y = [1, 2]

    with pytest.raises(ValueError, match="Os dados de x e y devem ter o mesmo comprimento."):
        regressao(x, y, grau=1)

def test_regressao_linear_ajuste_perfeito():
    """Testa uma regressão linear perfeita: y = 2x + 1"""
    x = [0, 1, 2]
    y = [1, 3, 5]
    esperado_a, esperado_b = 2.0, 1.0
    
    a, b = regressao_linear(x, y)
    
    assert a == pytest.approx(esperado_a, abs=TOLERANCIA)
    assert b == pytest.approx(esperado_b, abs=TOLERANCIA)

def test_regressao_linear_error_comprimentos_diferentes():
    """Testa se levanta ValueError para comprimentos de x e y diferentes."""
    x = [1, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError, match="X e y devem ter o mesmo número de amostras."):
        regressao_linear(x, y)

def test_regressao_logaritmica_ajuste_perfeito():
    """Testa uma regressão logarítmica perfeita: y = 2*ln(x) + 3"""
    # Usamos math.e para que np.log(math.e) == 1.0
    x = [1, math.e, math.e**2] 
    y = [3, 5, 7]
    
    a, b = regressao_logaritmica(x, y)
    
    assert a == pytest.approx(2.0, abs=TOLERANCIA)
    assert b == pytest.approx(3.0, abs=TOLERANCIA)

def test_regressao_logaritmica_error_x_negativo():
    """Testa se levanta ValueError se x contém valores negativos."""
    x = [1, -1, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError, match="Todos os valores de 'x' devem ser positivos"):
        regressao_logaritmica(x, y)

def test_regressao_logaritmica_error_x_zero():
    """Testa se levanta ValueError se x contém zero."""
    x = [1, 0, 2]
    y = [1, 2, 3]

    with pytest.raises(ValueError, match="Todos os valores de 'x' devem ser positivos"):
        regressao_logaritmica(x, y)

def test_polinomio_de_taylor_exp_x_em_0(x_symbol):
    """Testa P_3(x) para f(x) = e^x em a=0. Esperado: 1 + x + x^2/2 + x^3/6"""
    func = sp.exp(x_symbol)
    point = 0
    times = 4 # Grau 3 (termos de 0 a 3)
    
    expected_poly = 1 + x_symbol + x_symbol**2 / 2 + x_symbol**3 / 6
    
    poly = polinomio_de_taylor(func, x_symbol, point, times, plot=False)
    
    # Compara a forma expandida para garantir a igualdade simbólica
    assert poly.expand() == expected_poly.expand()

def test_polinomio_de_taylor_sin_x_em_0(x_symbol):
    """Testa P_3(x) para f(x) = sin(x) em a=0. Esperado: x - x^3/6"""
    func = sp.sin(x_symbol)
    point = 0
    times = 4 # Grau 3 (termos 0, 1, 2, 3)
    
    expected_poly = x_symbol - x_symbol**3 / 6
    
    poly = polinomio_de_taylor(func, x_symbol, point, times, plot=False)
    
    assert poly.expand() == expected_poly.expand()

def test_polinomio_de_taylor_cos_x_em_pi(x_symbol):
    """Testa P_2(x) para f(x) = cos(x) em a=pi. Esperado: -1 + (x-pi)^2/2"""
    func = sp.cos(x_symbol)
    point = sp.pi
    times = 3 # Grau 2 (termos 0, 1, 2)
    
    expected_poly = -1 + (x_symbol - sp.pi)**2 / 2
    
    poly = polinomio_de_taylor(func, x_symbol, point, times, plot=False)
    
    assert poly.expand() == expected_poly.expand()