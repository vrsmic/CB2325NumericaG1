from CB2325NumericaG1.interpolacao import lin_interp, hermite_interp, poly_interp
import pytest


def test_lin_interp():
    p = lin_interp([0, 1, 2, 3], [1, 2, 0, 4])
    assert p(1.5) == 1.0
    assert p(2.75) == 3.0
    assert p(0.2) == 1.2

def test_hermite_interp():
    p = hermite_interp([3, 0], [1, 4], [0, 0])
    assert p(1.5) == 2.5

def test_poly_interp():
    # testes se a função retorna valores numéricos esperados
    x = [0, 1, 2]
    y = [1, 3, 2]
    p = poly_interp(x, y)

    assert p(1.5) == pytest.approx(2.875)
    assert p(2.75) == pytest.approx(-0.71875)
    assert p(0.2) == pytest.approx(1.64)
    # exemplo do PDF de instruções
    x_pdf = [0, 1, 2, 3]
    y_pdf = [1, 2, 0, 4]
    p = poly_interp(x_pdf, y_pdf)
    assert p(1.5) == pytest.approx(0.8125)

    # testes de tratamento de erros
    # valores de x com duplicatas
    x_dup = [0, 1, 2, 2]
    y_dup = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        poly_interp(x_dup, y)
    
    # listas com tamanhos diferentes
    x_diff = [0, 1, 2]
    y_diff = [1, 2, 3, 4]
    with pytest.raises(ValueError):
        poly_interp(x_diff, y_diff)
    
    # listas vazias
    x_empty = []
    y_empty = []
    with pytest.raises(ValueError):
        poly_interp(x_empty, y_empty)