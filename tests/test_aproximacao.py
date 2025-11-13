# Python 3

# Bibliotecas padrão
import math

# Bibliotecas de terceiros
import pytest
import numpy as np
import numpy.testing as npt
import sympy as sp

# Funções a serem testadas (do outro arquivo)
from CB2325NumericaG1.aproximacao import regressao, ajuste_linear, regressao_logaritmica, polinomio_de_taylor, ajuste_trigonometrico

TOLERANCIA = 1e-6 # Tolerância para comparações de float.

@pytest.fixture
def sym_x():
    return sp.symbols('x')


### 1. Testes para regressao()
def test_regressao_grau_1_perfeita():
    # y = 2x + 3
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 5, 7, 9])
    
    # Espera [c0, c1] -> [3, 2]
    coefs = regressao(x, y, grau=1, plot=False)
    
    assert len(coefs) == 2
    npt.assert_allclose(coefs, [3.0, 2.0], atol=1e-9)

def test_regressao_grau_2_perfeita():
    # y = 1*x^2 + 2*x + 3
    x = np.array([0, 1, 2, 3])
    y = np.array([3, 6, 11, 18])
    
    # Espera [c0, c1, c2] -> [3, 2, 1]
    coefs = regressao(x, y, grau=2, plot=False)
    
    assert len(coefs) == 3
    npt.assert_allclose(coefs, [3.0, 2.0, 1.0], atol=1e-9)



### 2. Testes para regressao_logaritmica()
def test_regressao_logaritmica_perfeita():
    # y = 2 * ln(x) + 3
    x = np.array([1, np.e, np.e**2, np.e**3])
    # ln(x) = [0, 1, 2, 3]
    y = np.array([3, 5, 7, 9])
    
    a, b = regressao_logaritmica(x, y, plot=False)
    
    npt.assert_allclose([a, b], [2.0, 3.0], atol=1e-9)

def test_regressao_logaritmica_erro_x_zero():
    x = np.array([1, 2, 0])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="positivos"):
        regressao_logaritmica(x, y)

def test_regressao_logaritmica_erro_x_negativo():
    x = np.array([1, 2, -1])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="positivos"):
        regressao_logaritmica(x, y)

### 3. Testes para polinomio_de_taylor()
def test_taylor_exp_x_em_zero(sym_x):
    # Série de e^x em x=0
    # 1 + x + x^2/2
    f = sp.exp(sym_x)
    poly = polinomio_de_taylor(f, sym_x, 0, 3, plot=False)
    
    esperado = 1 + sym_x + sym_x**2 / 2
    assert poly == esperado

def test_taylor_sin_x_em_zero(sym_x):
    # Série de sin(x) em x=0
    # x - x^3/6
    f = sp.sin(sym_x)
    poly = polinomio_de_taylor(f, sym_x, 0, 4, plot=False)
    
    esperado = sym_x - sym_x**3 / 6
    assert poly == esperado

def test_taylor_ln_x_em_um(sym_x):
    # Série de ln(x) em x=1
    # (x-1) - (x-1)^2/2
    f = sp.log(sym_x)
    poly = polinomio_de_taylor(f, sym_x, 1, 3, plot=False)
    
    esperado = (sym_x - 1) - (sym_x - 1)**2 / 2
    assert poly.expand() == esperado.expand()

def test_taylor_plot_executa(sym_x, capsys):
    # Testa se a função 'plot' imprime a mensagem esperada
    f = sp.exp(sym_x)
    polinomio_de_taylor(f, sym_x, 0, 2, plot=True)
    
    captured = capsys.readouterr()
    assert "Gerando gráfico de comparação" in captured.out

### 4. Testes para ajuste_trigonometrico()

from CB2325NumericaG1.aproximacao import ajuste_trigonometrico

def test_ajuste_trigonometrico_perfeito_misto():
    T = 4.0
    omega = np.pi / 2.0
    
    x = np.array([0, 1, 2, 3])
    
    y = np.array([
        2.0 + 3.0*np.cos(omega*0) + 1.5*np.sin(omega*0), 
        2.0 + 3.0*np.cos(omega*1) + 1.5*np.sin(omega*1), 
        2.0 + 3.0*np.cos(omega*2) + 1.5*np.sin(omega*2), 
        2.0 + 3.0*np.cos(omega*3) + 1.5*np.sin(omega*3)  
    ])
    
    c0, c1, c2 = ajuste_trigonometrico(x, y, periodo=T, plot=False)
    
    npt.assert_allclose([c0, c1, c2], [2.0, 3.0, 1.5], atol=TOLERANCIA)

def test_ajuste_trigonometrico_erro_poucos_pontos():
    x = np.array([1, 2])
    y = np.array([5, 4])
    T = 10.0
    
    with pytest.raises(ValueError, match="pelo menos 3 pontos"):
        ajuste_trigonometrico(x, y, periodo=T)

def test_ajuste_trigonometrico_erro_periodo_invalido():
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 1, 2])
    
    with pytest.raises(ValueError, match="período deve ser um valor positivo"):
        ajuste_trigonometrico(x, y, periodo=0)
        
    with pytest.raises(ValueError, match="período deve ser um valor positivo"):
        ajuste_trigonometrico(x, y, periodo=-2.0)