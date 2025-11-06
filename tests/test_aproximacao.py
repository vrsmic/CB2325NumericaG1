import CB2325NumericaG1.integracao

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def test_regressao_linear_reta_perfeita():
    """Testa a regressão com uma reta perfeita y = 2x + 1."""
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


@pytest.fixture
def x_symbol():
    """Fixture para fornecer o símbolo 'x' do sympy."""
    return sp.Symbol('x')

def test_polinomio_de_taylor_exp_x_em_0(x_symbol):
    """Testa P_3(x) para f(x) = e^x em a=0. Esperado: 1 + x + x^2/2 + x^3/6"""
    func = sp.exp(x_symbol)
    point = 0
    times = 4 # Grau 3
    
    expected_poly = 1 + x_symbol + x_symbol**2 / 2 + x_symbol**3 / 6
    
    poly = polinomio_de_taylor(func, x_symbol, point, times)
    
    # Compara a forma expandida para garantir a igualdade simbólica
    assert poly.expand() == expected_poly.expand()

def test_polinomio_de_taylor_sin_x_em_0(x_symbol):
    """Testa P_3(x) para f(x) = sin(x) em a=0. Esperado: x - x^3/6"""
    func = sp.sin(x_symbol)
    point = 0
    times = 4 # Grau 3
    expected_poly = x_symbol - x_symbol**3 / 6
    poly = polinomio_de_taylor(func, x_symbol, point, times)
    assert poly.expand() == expected_poly.expand()