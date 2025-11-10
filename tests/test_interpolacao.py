from CB2325NumericaG1.interpolacao import lin_interp, hermite_interp, poly_interp, vandermond_interp
from pytest import approx, raises

def test_lin_interp():
    """Teste da função lin_interp.
    
    Verificação da linearidade da função em cada intervalo de interpolação.
    Logo, se x em (1,2), y varia de 2 a 0, então para 1.5 (meio do caminho), o
    valor esperado é 2/2 = 1.0; e assim por diante.
    """
    p = lin_interp([0, 1, 2, 3], [1, 2, 0, 4])
    assert p(1.5) == approx(1.0)
    assert p(2.75) == approx(3.0)
    assert p(0.2) == approx(1.2)

def test_hermite_interp():
    """Teste da função hermite_interp.

    Teste da função y = x^2. Esperado que p(x) == x^2. Note que p'(x) = 2*x
    (terceira entrada de hermite_interp)
    """
    p = hermite_interp([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4], [-4, -2, 0, 2, 4])
    assert p(3) == approx(9)
    assert p(2.5) == approx(6.25)

def test_poly_interp():
    """Teste da função poly_interp.
    
    Teste da função padrão dada no documento Trabalho_de_Grupo_cb23_2025.pdf
    """
    # testes se a função retorna valores numéricos esperados
    x = [0, 1, 2]
    y = [1, 3, 2]
    p = poly_interp(x, y)

    assert p(1.5) == approx(2.875)
    assert p(2.75) == approx(-0.71875)
    assert p(0.2) == approx(1.64)

    # exemplo do PDF de instruções
    x_pdf = [0, 1, 2, 3]
    y_pdf = [1, 2, 0, 4]
    p = poly_interp(x_pdf, y_pdf)
    assert p(1.5) == approx(0.8125)

    # testes de tratamento de erros
    # valores de x com duplicatas
    x_dup = [0, 1, 2, 2]
    y_dup = [1, 2, 3, 4]
    with raises(ValueError):
        poly_interp(x_dup, y_dup)
    
    # listas com tamanhos diferentes
    x_diff = [0, 1, 2]
    y_diff = [1, 2, 3, 4]
    with raises(ValueError):
        poly_interp(x_diff, y_diff)
    
    # listas vazias
    x_empty = []
    y_empty = []
    with raises(ValueError):
        poly_interp(x_empty, y_empty)

def test_vandermond_interp():
    """Teste da função vandermond_interp.
    
    Teste da função y = x^2. Esperado que p(x) == x^2.
    """
    p = vandermond_interp([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])
    assert p(3) == approx(9)
    assert p(2.5) == approx(6.25)