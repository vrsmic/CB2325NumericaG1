from CB2325NumericaG1.interpolacao import lin_interp, hermite_interp

def test_lin_interp():
    p = lin_interp([0, 1, 2, 3], [1, 2, 0, 4])
    assert p(1.5) == 1.0
    assert p(2.75) == 3.0
    assert p(0.2) == 1.2

def test_hermite_interp():
    p = hermite_interp([3, 0], [1, 4], [0, 0])
    assert p(1.5) == 2.5