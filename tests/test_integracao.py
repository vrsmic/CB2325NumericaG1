import CB2325NumericaG1.integracao
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