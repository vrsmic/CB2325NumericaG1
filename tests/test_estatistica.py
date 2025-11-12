import numpy as np
from pytest import approx
import pytest


from CB2325NumericaG1.estatistica import mean, std



def test_mean_simples_inteiros():

    dados = [1, 2, 3]
    assert mean(dados) == approx(2.0)

def test_mean_simples_floats():

    dados = [1.5, 2.5, 3.5]
    assert mean(dados) == approx(2.5)

def test_mean_um_valor():

    dados = [42]
    assert mean(dados) == approx(42.0)

def test_mean_ponderada_simples():

    dados = [1, 3]
    pesos = [1, 1]
    assert mean(dados, pesos) == approx(2.0)

def test_mean_ponderada_diferente():

    # (10*1 + 20*3) / (1 + 3) = 70 / 4 = 17.5
    dados = [10, 20]
    pesos = [1, 3]
    assert mean(dados, pesos) == approx(17.5)

# --- Testes para a função std ---


def test_std_sem_desvio():

    dados = [5, 5, 5, 5]
    assert std(dados) == approx(0.0)

def test_std_simples_inteiros():
    # Testa o desvio padrão de [1, 3]. Média=2.

    dados = [1, 3]
    assert std(dados) == approx(1.0)

def test_std_com_negativos():
    # Testa o desvio padrão com números negativos. Média=0.

    dados = [-1, 1]
    assert std(dados) == approx(1.0)

def test_std_lista_tres_elementos():
    # Var = ((1-2)**2 + (2-2)**2 + (3-2)**2) / 3 = (1 + 0 + 1) / 3 = 2/3.
    dados = [1, 2, 3]
    assert std(dados) == approx((2/3)**0.5)

def test_std_lista_floats():
    # Var = ((1.5-2.5)**2 + (2.5-2.5)**2 + (3.5-2.5)**2) / 3 = (1 + 0 + 1) / 3 = 2/3.
    dados = [1.5, 2.5, 3.5]
    assert std(dados) == approx((2/3)**0.5)