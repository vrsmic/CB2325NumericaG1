from pytest import approx
from CB2325NumericaG1.erros import erro_absoluto, erro_relativo

def test_erro_absoluto_simples1():
    # Testa se o erro absoluto entre números naturias, 4 e 3, é 1.
    assert erro_absoluto(4,3) == 1

def test_erro_absoluto_inverso1():
    # Testa se o erro absoluto entre números naturais, 3 e 4, é 1.
    assert erro_absoluto(3,4) == 1

def test_erro_absoluto_decimal1():
    # Testa se o erro absoluto entre 3.4567 e 5.456 é aproximadamente 1.9993
    assert erro_absoluto(3.4567, 5.456) == approx(1.9993)

def test_erro_relativo_simples1():
    # Testa se o erro relativo entre 4 (valor real) e 2 (valor aproximado) é aproximadamente 0.25.
    assert erro_relativo(4, 2) == approx(0.5)

def test_erro_relativo_inverso1():
    # Testa se o erro relativo entre 2 (valor real) e 4 ( valor aproximado) é aproximadamente 1.
    assert erro_relativo(2, 4) == approx(1)

def test_erro_relativo_decimal1():
     # Testa se o erro relativo entre 2.25 (valor real) e 4.75 ( valor aproximado) é aproximadamente 1.11...
    assert erro_relativo(2.25, 4.75) == approx(2.5 / 2.25)
