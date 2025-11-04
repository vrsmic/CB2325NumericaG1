from CB2325NumericaG1.integracao import trapezio, simpson13
import math

def test_trapezio_seno() :
    
    """Testa a integral de sin(x) de 0 a pi.
        Resultado aproximadamente 2."""
    
    f = lambda x: math.sin(x)
    resultado = trapezio(f, 0, math.pi, n=100)
    assert abs(resultado - 2.0) < 0.01

def test_trapezio_quadrado() :

    """Testa a integral de x² de 0 a 1.
        Resultado aproximadamente 1/3."""
    
    f = lambda x: x**2
    resultado = trapezio(f, 0, 1, n = 100)
    assert abs(resultado - (1/3)) < 0.01

def test_trapezio_constante() :

    """Testa a integral de f(x) = 5 de 0 a 2 = 10.
        Resultado aproximadamente 10."""
    
    f = lambda x: 5
    resultado = trapezio(f, 0, 2, n = 10)
    assert abs(resultado - 10) < 0.001

def test_trapezio_negativa() :

    """Testa f(x) = -x de 0 a 2.
        Resultado aproximadamente -2."""
    
    f = lambda x: -x
    resultado = trapezio(f, 0, 2, n = 50)
    assert abs(resultado + 2) < 0.01



def test_simpson13_seno() :
    
    """Testa a integral de sin(x) de 0 a pi.
        Resultado aproximadamente 2."""
    
    f = lambda x: math.sin(x)
    resultado = simpson13(f, 0, math.pi, n=100)
    assert abs(resultado - 2.0) < 0.01

def test_simpson13_quadrado() :

    """Testa a integral de x² de 0 a 1.
        Resultado aproximadamente 1/3."""
    
    f = lambda x: x**2
    resultado = simpson13(f, 0, 1, n = 100)
    assert abs(resultado - (1/3)) < 0.01

def test_simpson13_constante() :

    """Testa a integral de f(x) = 5 de 0 a 2 = 10.
        Resultado aproximadamente 10."""
    
    f = lambda x: 5
    resultado = simpson13(f, 0, 2, n = 10)
    assert abs(resultado - 10) < 0.001

def test_simpson13_negativa() :

    """Testa f(x) = -x de 0 a 2.
        Resultado aproximadamente -2."""
    
    f = lambda x: -x
    resultado = simpson13(f, 0, 2, n = 50)
    assert abs(resultado + 2) < 0.01

# Integração estocástica

def test_monte_carlo_one_variable_seno():
    """Testa a integral de sin(x) de 0 a pi pelo método de Monte Carlo.
       Resultado esperado aproximadamente 2."""
    
    f = lambda x: math.sin(x)
    resultado = monte_carlo_one_variable(f, 0, math.pi, n=100000)
    assert abs(resultado - 2.0) < 0.05


def test_monte_carlo_one_variable_quadrado():
    """Testa a integral de x² de 0 a 1 pelo método de Monte Carlo.
       Resultado esperado aproximadamente 1/3."""
    
    f = lambda x: x**2
    resultado = monte_carlo_one_variable(f, 0, 1, n=100000)
    assert abs(resultado - (1/3)) < 0.05


def test_monte_carlo_one_variable_constante():
    """Testa a integral de f(x) = 5 de 0 a 2 = 10 pelo método de Monte Carlo.
       Resultado esperado aproximadamente 10."""
    
    f = lambda x: 5
    resultado = monte_carlo_one_variable(f, 0, 2, n=50000)
    assert abs(resultado - 10) < 0.05


def test_monte_carlo_one_variable_negativa():
    """Testa f(x) = -x de 0 a 2 pelo método de Monte Carlo.
       Resultado esperado aproximadamente -2."""
    
    f = lambda x: -x
    resultado = monte_carlo_one_variable(f, 0, 2, n=100000)
    assert abs(resultado + 2) < 0.05


def test_monte_carlo_two_variables_x2y():
    """Testa a integral dupla de f(x,y) = x²*y em [0,1]x[0,1].
       Resultado esperado: ∫₀¹∫₀¹ x²y dy dx = 1/6 ≈ 0.1667."""
    
    f = lambda x, y: (x**2) * y
    resultado = monte_carlo_two_variables(f, 0, 1, 0, 1, n=200000)
    assert abs(resultado - (1/6)) < 0.05


def test_monte_carlo_two_variables_soma():
    """Testa a integral dupla de f(x,y) = x + y em [0,1]x[0,1].
       Resultado esperado: ∫₀¹∫₀¹ (x + y) dy dx = 1.0."""
    
    f = lambda x, y: x + y
    resultado = monte_carlo_two_variables(f, 0, 1, 0, 1, n=200000)
    assert abs(resultado - 1.0) < 0.05


def test_monte_carlo_two_variables_constante():
    """Testa a integral dupla de f(x,y) = 3 em [0,2]x[0,1].
       Resultado esperado: 3 * área = 3 * 2 * 1 = 6."""
    
    f = lambda x, y: 3
    resultado = monte_carlo_two_variables(f, 0, 2, 0, 1, n=150000)
    assert abs(resultado - 6) < 0.05


def test_monte_carlo_two_variables_negativa():
    """Testa a integral dupla de f(x,y) = -x*y em [0,1]x[0,1].
       Resultado esperado: ∫₀¹∫₀¹ -x*y dy dx = -1/4 = -0.25."""
    
    f = lambda x, y: -x * y
    resultado = monte_carlo_two_variables(f, 0, 1, 0, 1, n=200000)
    assert abs(resultado + 0.25) < 0.05
