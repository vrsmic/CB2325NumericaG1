import numpy as np
def hermite_interp(x_pontos: list, y_pontos: list, dy_pontos: list, allow_extrapolation: bool = False) -> object | None:
    """
    Cria uma função de interpolação polinomial de Hermite.
    
    Retorna uma função (do tipo 'object' para fins de anotação) 
    ou None se der erro.

    Args:
        x_pontos: Coordenadas x (n valores).
        y_pontos: Coordenadas y (n valores).
        dy_pontos: Derivadas dy/dx em cada x (n valores).
        allow_extrapolation: Se False (padrão), levanta um ValueError 
                                 para valores de x fora do intervalo de dados.

    Raises:
        ValueError: Se `allow_extrapolation` for False e `x_novo`
                    estiver fora do intervalo [min(x_pontos), max(x_pontos)].
    Notas:
        Sobre a Extrapolação:
        Por padrão, esta biblioteca não permite extrapolação (allow_extrapolation=False).
        Isso evita que o polinômio seja avaliado em regiões onde ele
        tende a crescer rapidamente e perder precisão numérica.
    """
    try:
        x_pts = np.asarray(x_pontos, dtype=float)
        y_pts = np.asarray(y_pontos, dtype=float)
        dy_pts = np.asarray(dy_pontos, dtype=float)
    except Exception as e:
        print(f"Erro ao converter entradas para arrays numpy: {e}")
        return None

    n = len(x_pts)
    if n == 0:
        print("Erro: As listas de pontos não podem estar vazias.")
        return None
    if len(y_pts) != n or len(dy_pts) != n:
        print("Erro: As listas x, y, e dy devem ter o mesmo tamanho.")
        return None
        
    num_coefs = 2 * n
    
    x_min = np.min(x_pts)
    x_max = np.max(x_pts)
    
    A = np.zeros((num_coefs, num_coefs))
    b = np.zeros(num_coefs)
    
    for i in range(n):
        x = x_pts[i]
        
        A[2*i] = [x**j for j in range(num_coefs)]
        b[2*i] = y_pts[i]
        
        linha_dy = [0.0] + [j * x**(j-1) for j in range(1, num_coefs)]
        A[2*i + 1] = linha_dy
        b[2*i + 1] = dy_pts[i]
    
    try:
        coefs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Erro: Matriz singular. Verifique se há pontos x duplicados.")
        return None

    def polinomio_interpolador_hermite(x_novo: float | np.ndarray) -> float | np.ndarray:
        """
        Avalia o polinômio P(x_novo) = sum(c_j * x_novo^j)
        """
        x_val = np.asarray(x_novo, dtype=float)
        
        if not allow_extrapolation:
            out_of_bounds = (x_val < x_min) | (x_val > x_max)
            if np.any(out_of_bounds):
                raise ValueError(
                    f"Valores {x_val[out_of_bounds]} estão fora do intervalo de interpolação [{x_min}, {x_max}]. "
                    "Use allow_extrapolation=True para forçar o cálculo."
                )
                        
        return np.polyval(coefs[::-1], x_val)
    
    return polinomio_interpolador_hermite