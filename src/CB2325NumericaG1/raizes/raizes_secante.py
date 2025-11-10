import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

def secante(function: Callable, guess0: float, guess1: float, tolerance: float, plot: bool = False) -> float:
    """
    Encontra/aproxima uma raiz de uma função real de variável real usando o método da secante.

    Calcula onde a reta secante ao gráfico da função nos pontos guess0 e guess1 cruza o eixo x.
    Repete o processo com esse novo ponto guess2 e o ponto guess1.
    
    Args:
        function (Callable):
            Função cuja raíz queremos encontrar ou aproximar.
        guess0 (float):
            Primeiro chute inicial.
        guess1 (float):
            Segundo chute inicial.
        tolerance (float):
            Critério de parada.
            O método para quando |f| < tolerance ou |dx| < tolerance.
        plot (bool = False):
            Determina se uma visualização gráfica do método será plotada.
            Por padrão, não será.

    Returns:
        float:
            Valor aproximado da raiz.

    Raises:
        ValueError:
            Se ocorrer NaN/Inf em algum momento da iteração.
        TypeError:
            Se function não for Callable.
        ZeroDivisionError:
            Se a derivada praticamente zerar em algum momento da iteração.
        RunTimeError:
            Se o método não convergir em no máximo 1000 iterações.
    """
    
    if not callable(function):
        raise TypeError("function deve ser um Callable.")

    f = function

    x_prev = float(guess0)
    x_curr = float(guess1)
    
    f_prev = float(f(x_prev))
    
    if np.isnan(f_prev) or np.isinf(f_prev):
        raise ValueError(f"f(x) retornou {f_prev} no chute inicial x0 = {x_prev}.")
    
    x_record = [x_prev, x_curr]
    
    root = None
    MAX_ITERS = 1000
    
    for i in range(MAX_ITERS):
        f_curr = float(f(x_curr))
        
        if np.isnan(f_curr) or np.isinf(f_curr):
            raise ValueError(f"f(x) retornou {f_curr} no ponto x = {x_curr} na iteração {i}.")
        
        # Critério de parada
        if abs(f_curr) < tolerance:
            root = x_curr
            break
        
        if abs(f_curr - f_prev) < 1e-16:
            raise ZeroDivisionError(f"Denominador muito próximo de zero na iteração {i}.")
        
        # A fórmula de iteração da Secante
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        x_record.append(x_next)
        
        # Critério de parada
        if abs(x_next - x_curr) < tolerance:
            root = x_next
            break
            
        # Atualiza os pontos para a próxima iteração
        x_prev = x_curr
        x_curr = x_next
        f_prev = f_curr
    
    if root is None:
        f_last = float(f(x_curr))
        raise RuntimeError(f"Não convergiu após {MAX_ITERS} iterações. Último x = {x_curr}, f(x) = {f_last}")
    
    if plot:
        pass
    
    return root