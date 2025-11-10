from CB2325NumericaG1.interpolacao import lin_interp, hermite_interp, poly_interp, vandermond_interp
from pytest import approx

def test_lin_interp():
    p = lin_interp([0, 1, 2, 3], [1, 2, 0, 4])
    assert p(1.5) == approx(1.0)
    assert p(2.75) == approx(3.0)
    assert p(0.2) == approx(1.2)

def test_hermite_interp():
    p = hermite_interp([3, 0], [1, 4], [0, 0])
    assert p(1.5) == approx(2.5)

def test_vandermond_interp():
    p = vandermond_interp([-2, -1, 0, 1, 2], [4, 1, 0, 1, 4])
    assert p(3) == approx(9)
    assert p(2.5) == approx(6.25)