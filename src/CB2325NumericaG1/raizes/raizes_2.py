import numpy as np
import sympy as sp

def newton_raphson(function, guess, tolerance):
    """
    Encontra uma raiz de uma função real usando o método de Newton–Raphson.
    
    Parâmetros
    ----------
    function : callable ou sympy.Expr/sympy.Lambda
        Função cujo zero queremos encontrar. Deve aceitar um único argumento x (float).
        Pode ser:
         - um callable Python (por exemplo, lambda x: x**2 - 2) que usa operações com floats/numpy, ou
         - uma expressão SymPy (por exemplo, sp.sympify("x**2 - 2")) ou sp.Lambda.
    guess : float
        Chute inicial x0.
    tolerance : float
        Critério de parada. O algoritmo para quando |f(x)| < tolerance ou |dx| < tolerance.
    
    Retorna
    -------
    float
        Aproximação da raiz.
    
    Levanta
    ------
    ZeroDivisionError se a derivada for (praticamente) zero durante a iteração.
    ValueError se a função produzir NaN/Inf no chute.
    RuntimeError se não convergir dentro de um número máximo de iterações.
    """
    MAX_ITERS = 1000
    x0 = float(guess)
    
    # Preparar f e df: se function for simbólica, obtenha derivada simbólica;
    # caso contrário, use derivada numérica por diferença central.
    if isinstance(function, sp.Basic):  # cobre sp.Expr, sp.Symbol, etc.
        if isinstance(function, sp.Lambda):
            f_expr = function.expr
            variables = function.variables
        else: # é uma expressão sp.Basic
            f_expr = function
            variables = function.free_symbols

        # Garante que é uma função de UMA variável
        if len(variables) == 0:
            # É uma constante, ex: sp.sympify("5")
            f = lambda x: float(f_expr)
            df = lambda x: 0.0
        elif len(variables) == 1:
            x_sym = list(variables)[0]
            f = sp.lambdify(x_sym, f_expr, 'numpy')
            
            df_expr = sp.diff(f_expr, x_sym)
            df = sp.lambdify(x_sym, df_expr, 'numpy')
        else:
            raise ValueError(f"A expressão SymPy deve ter exatamente uma variável, mas foram encontradas {len(variables)}: {variables}")
    

    elif callable(function):
        f = function
        def df(x):
            h = 1e-8
            return (f(x + h) - f(x - h)) / (2.0 * h)
    
    # Caso 3: Entrada inválida
    else:
        raise TypeError("function deve ser um callable (ex: lambda) ou uma expressão SymPy (Expr ou Lambda).")
    
    for i in range(MAX_ITERS):
        fx = float(f(x0))
        if np.isnan(fx) or np.isinf(fx):
            raise ValueError(f"f(x) retornou {fx} no ponto x = {x0}.")
        # critério pelo valor da função
        if abs(fx) < tolerance:
            return x0
        dfx = float(df(x0))
        if np.isnan(dfx) or np.isinf(dfx):
            raise ValueError(f"f'(x) retornou {dfx} no ponto x = {x0}.")
        if abs(dfx) < 1e-16:
            raise ZeroDivisionError(f"Derivada muito próxima de zero em x = {x0}.")
        x1 = x0 - fx / dfx
        # critério pelo passo
        if abs(x1 - x0) < tolerance:
            return x1
        x0 = x1
    
    raise RuntimeError(f"Não convergiu após {MAX_ITERS} iterações. Último x = {x0}, f(x) = {fx}")
