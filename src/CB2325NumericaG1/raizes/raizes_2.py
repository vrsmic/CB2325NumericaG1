import numpy as np
import sympy as sp
import matplotlib as plt
from typing import Callable, Union

def newton_raphson(function: Union[Callable, sp.Basic], guess: float, tolerance: float) -> float:
    """
    Encontra/aproxima uma raiz de uma função real de variável real usando o método de Newton–Raphson.

    Calcula onde a reta tangente ao gráfico da função no ponto x0 cruza o eixo x.
    Repete o processo com esse novo ponto x1.
    
    Args:
        function Union[Callable, sp.Basic]:
            Função cuja raíz queremos encontrar ou aproximar.
            Pode ser Callable ou sp.Basic.
            Para melhor eficiência do método, deve ser sp.Basic.
        guess (float):
            Chute inicial x0.
        tolerance (float):
            Critério de parada.
            O método para quando |f(x_n)| < tolerance ou |x_{n+1} - x_n| < tolerance.

    Returns:
        float:
            Valor aproximado da raiz, arredondado para 4 casas decimais.

    Raises:
        ValueError:
            Se a expressão SymPy tiver mais de uma variável.
            Se ocorrer NaN/Inf em algum momento da iteração.
        TypeError:
            Se function não for Callable ou sp.Basic.
        ZeroDivisionError:
            Se a derivada praticamente zerar em algum momento da iteração.
        RunTimeError:
            Se o método não convergir em no máximo 1000 iterações.
    """

    MAX_ITERS = 1000
    x0 = float(guess)
    x_record = [x0]

    # Preparar f e df: se function for sympy, obtenha a derivada analítca;
    # caso contrário, use derivada numérica por quociente de newton.
    if isinstance(function, sp.Basic):  # cobre sp.Expr, sp.Symbol, etc.
        if isinstance(function, sp.Lambda):
            f_expr = function.expr
            variables = function.variables
        else: # é uma expressão sp.Basic
            f_expr = function
            variables = function.free_symbols

        # Garante que é uma função de uma variável
        if len(variables) == 0:
            # Caso constante
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
        raise TypeError("function deve ser Callable ou sp.Basic.")
    
    root = None
    for i in range(MAX_ITERS):
        fx = float(f(x0))
        if np.isnan(fx) or np.isinf(fx):
            raise ValueError(f"f(x) retornou {fx} no ponto x = {x0}.")
        # critério pelo valor da função
        if abs(fx) < tolerance:
            root = x0
            break
        dfx = float(df(x0))
        if np.isnan(dfx) or np.isinf(dfx):
            raise ValueError(f"f'(x) retornou {dfx} no ponto x = {x0}.")
        if abs(dfx) < 1e-16:
            raise ZeroDivisionError(f"Derivada muito próxima de zero em x = {x0}.")
        x1 = x0 - fx / dfx
        x_record.append(x1)
        # critério pelo passo
        if abs(x1 - x0) < tolerance:
            root = x1
            break
        x0 = x1
    
    if root is None:
        raise RuntimeError(f"Não convergiu após {MAX_ITERS} iterações. Último x = {x0}, f(x) = {fx}")
    
    # Visualização gráfica
    if len(x_record) > 1:
        x_min = min(x_record)
        x_max = max(x_record)
        delta = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
        x_min -= delta
        x_max += delta
    else:
        x_min = x_record[0] - 1.0
        x_max = x_record[0] + 1.0
    
    x = np.linspace(x_min, x_max, 100)
    y = f(x)
    
    # Plota a função original em preto.
    plt.plot(x, y, color='black', linewidth=1, label='f(x)')
    
    # Plota os pontos de iteração
    x_points_y = f(np.array(x_record))
    plt.plot(x_record, x_points_y, 'ro', label='Pontos de Iteração')
    
    # Plota os segmentos de reta tangente pontilhados
    for j in range(len(x_record) - 1):
        x_i = x_record[j]
        y_i = float(f(x_i))
        x_next = x_record[j + 1]
        x_tang = np.array([x_i, x_next])
        y_tang = df(x_i) * (x_tang - x_i) + y_i
        plt.plot(x_tang, y_tang, 'b--', label='Tangente' if j == 0 else None)
    
    # Plota o eixo x.
    plt.axhline(0, color='black', linewidth=1)
    
    # Configuração do gráfico
    plt.axis('equal')
    plt.title("Raízes da função pelo Método de Newton-Raphson")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
    
    return round(root, 4)