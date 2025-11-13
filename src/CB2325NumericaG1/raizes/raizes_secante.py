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
            Se em algum momento da iteração o ponto xn nao estiver
            no domínio
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
    
    # Avalia f no primeiro chute com proteção
    try:
        f_prev_raw = f(x_prev)
        f_prev = float(f_prev_raw)
    except Exception as e:
        raise ValueError(f"f(x) não pôde ser avaliada no chute inicial x0 = {x_prev}: {e}")
    
    if np.isnan(f_prev) or np.isinf(f_prev):
        raise ValueError(f"f(x) retornou {f_prev} no chute inicial x0 = {x_prev}.")
    
    x_record = [x_prev, x_curr]
    MAX_ITERS = 1000
    root = None
    f_curr = None    

    for i in range(MAX_ITERS):
        # Avalia f no segundo chute com proteção
        try:
            f_curr_raw = f(x_curr)
            f_curr = float(f_curr_raw)
        except Exception as e:
            raise ValueError(f"f(x) não pôde ser avaliada no ponto x = {x_curr} na iteração {i}: {e}")

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
        try:
            f_last = float(f(x_curr))
        except Exception:
            f_last = float("nan")
        raise RuntimeError(f"Não convergiu após {MAX_ITERS} iterações. Último x = {x_curr}, f(x) = {f_last}")
    
    # Visualização Gráfica
    if plot:
        if len(x_record) > 1:
            x_min = min(x_record)
            x_max = max(x_record)
            delta = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
            x_min -= delta
            x_max += delta
        else:
            x_min = x_record[0] - 1.0
            x_max = x_record[0] + 1.0

        # Cria um range denso para plotar a curva da função
        x_space = np.linspace(x_min, x_max, 400)
        y_space = f(x_space)

        # Plota a função original
        plt.plot(x_space, y_space, color='black', linewidth=1.2, label='f(x)')

        # Pontos de iteração
        x_points = np.array(x_record)
        y_points = f(x_points)
        plt.plot(x_points, y_points, 'ro', label='Pontos de iteração', zorder=3)

        # Desenha as secantes
        for j in range(len(x_record) - 1):
            x0, y0 = x_record[j], f(x_record[j])
            x1, y1 = x_record[j + 1], f(x_record[j + 1])

            # Coeficiente angular e intercepto da secante
            m = (y1 - y0) / (x1 - x0)
            b = y1 - m * x1

            # Extensão da linha (até cruzar o eixo x)
            x_line = np.linspace(min(x0, x1), max(x0, x1), 10)
            y_line = m * x_line + b

            plt.plot(x_line, y_line, 'b--', linewidth=0.8, label='Secante' if j == 0 else None)

        # Plota o eixo x.
        plt.axhline(0, color='gray', linewidth=1)

        # Ajustes visuais
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title("Raízes da função pelo Método da Secante")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.show()
    
    return root