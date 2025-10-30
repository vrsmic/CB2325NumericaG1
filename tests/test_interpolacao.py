from CB2325NumericaG1.interpolacao import lin_interp

def test_lin_interp():
    p = lin_interp([0, 1, 2, 3], [1, 2, 0, 4])
    assert p(1.5) == 1.0