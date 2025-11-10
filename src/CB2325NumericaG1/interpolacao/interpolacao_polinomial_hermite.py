import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def _hermite_interp_mat(x_pontos: list, y_pontos: list, dy_pontos: list) -> Callable | None:
    """
    Função interna - Cria a função matemática da interpolação.

    Essa função apenas resolve o sistema linear e 
    retorna a função polinomial (Callable) que pode ser usada 
    para calcular valores.

    Parâmetros:
        x_pontos: Coordenadas x (n valores).
        y_pontos: Coordenadas y (n valores).
        dy_pontos: Derivadas dy/dx em cada x (n valores)."

    Retorna:
        Uma função (Callable) que avalia o polinômio, ou None se der erro.
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

    # valores min/max não são necessários pra checagem, mas podem ser úteis se quiser saber o intervalo
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

        return np.polyval(coefs[::-1], x_val)

    return polinomio_interpolador_hermite


def _ordenar_coordenadas_hermite(x: list, y: list, dy: list) -> tuple:
    """
    Função interna - Ordena as coordenadas mantendo 'pareamento' para Hermite.

    Parâmetros:
    x: lista das coordenadas x, em x[i], de cada ponto i.
    y: lista das coordenadas y, em y[i], de cada ponto i.
    dy: lista das derivadas dy[i] em cada ponto i.

    Retorna:
    x_ord: lista das coordenadas x em ordem crescente.
    y_ord: lista das coordenadas y, pareadas com as coordenadas x.
    dy_ord: lista das derivadas, pareadas com as coordenadas x.
    """
    x_np = np.array(x)
    y_np = np.array(y)
    dy_np = np.array(dy)

    idx = np.argsort(x_np)

    x_ord = x_np[idx]
    y_ord = y_np[idx]
    dy_ord = dy_np[idx]

    return x_ord, y_ord, dy_ord


def _plotar_hermite(x: list, y: list, dy: list, f: Callable, 
                    f_real: Callable | None = None,  
                    titulo: str = "Interpolação de Hermite"):
    """
    Função interna - Plotagem de pontos, derivadas e da função de interpolação.

    Parâmetros:
    x: lista das coordenadas x, em x[i], de cada ponto i.
    y: lista das coordenadas y, em y[i], de cada ponto i.
    dy: lista das derivadas dy[i] em cada ponto i.
    f: função de interpolação que será plotada.
    f_real: (Opcional) a função verdadeira para cálculo do erro.
    titulo: título do gráfico.

    Retorna:
    None
    """
    # criar pontos para a curva suave
    x_min, x_max = min(x), max(x)

    # estende ligeiramente o plot para mostrar a extrapolação, se desejado
    # você pode ajustar 'padding' ou remover se preferir plotar só o intervalo
    padding = 0.1 * (x_max - x_min)
    if padding == 0:  # caso de ponto único
        padding = 1.0

    x_curve = np.linspace(x_min - padding, x_max + padding, 500)
    y_curve = f(x_curve)

    # calcular as retas tangentes nos pontos de interpolação
    comprimento_tangente = 0.1 * \
        (x_max - x_min) if (x_max - x_min) > 0 else 0.1

    if f_real:
        # Se f_real for fornecida, cria 2 subplots (um para a função, um para o erro)
        fig, (ax, ax_err) = plt.subplots(
            2, 1, 
            figsize=(10, 10), 
            gridspec_kw={'height_ratios': [2, 1]},
            sharex=True # Compartilha o eixo x
        )
        
        # Plot da função real no gráfico principal (ax)
        y_real_curve = f_real(x_curve)
        ax.plot(x_curve, y_real_curve, 'k:', linewidth=2, label='Função Real')

        # Cálculo e plotagem do erro absoluto no gráfico inferior (ax_err)
        erro_absoluto = np.abs(y_real_curve - y_curve)
        ax_err.plot(x_curve, erro_absoluto, 'r-', label='Erro Absoluto |Real - Interp.|')
        ax_err.set_xlabel('x')
        ax_err.set_ylabel('Erro Absoluto')
        ax_err.set_title('Gráfico de Erro')
        ax_err.grid(True, alpha=0.3)
        ax_err.legend()
        
    else:
        # Comportamento original: apenas um gráfico
        fig, ax = plt.subplots(figsize=(10, 6))

    # gráfico principal
    ax.scatter(x, y, color='red', s=50, zorder=5, label='Pontos de dados')
    ax.plot(x_curve, y_curve, 'b-', linewidth=2, label='Polinômio de Hermite')

    # adc retas tangentes
    tangente_plotted = False
    for xi, yi, dyi in zip(x, y, dy):
        x_tang = [xi - comprimento_tangente, xi + comprimento_tangente]
        y_tang = [yi - comprimento_tangente *
                  dyi, yi + comprimento_tangente * dyi]
        label = 'Tangente' if not tangente_plotted else ""
        ax.plot(x_tang, y_tang, 'g--', alpha=0.7, linewidth=1, label=label)
        if not tangente_plotted:
            tangente_plotted = True
        ax.plot(xi, yi, 'ro', markersize=8)

    ax.set_ylabel('y')
    ax.set_title(titulo)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Se não houver gráfico de erro, adiciona o label 'x' ao gráfico principal
    if not f_real:
        ax.set_xlabel('x')

    plt.tight_layout()
    plt.show()


def hermite_interp(x_pontos: list, y_pontos: list, dy_pontos: list,
                   f_real: Callable | None = None,
                   titulo: str = "Interpolação de Hermite",
                   plot: bool = False) -> Callable:
    """
    Cria e plota uma função de interpolação polinomial de Hermite.

    Parâmetros:
        x_pontos: Coordenadas x (n valores).
        y_pontos: Coordenadas y (n valores).
        dy_pontos: Derivadas dy/dx em cada x (n valores).
        f_real: (Opcional) A função 'verdadeira' para plotar junto e calcular o gráfico de erro.
        titulo: Título para o gráfico.
        plot: indica se deve haver a plotagem (True) ou não (False).

    Retorna:
        Função de interpolação de Hermite.
        Se f_real for fornecida, imprime o erro médio e máximo

    Notas:
        Sobre a Extrapolação:
        Esta função sempre permite a extrapolação (avaliar valores de x 
        fora do intervalo de dados [min(x_pontos), max(x_pontos)]). 
        O usuário é responsável por verificar os resultados, pois 
        polinômios podem crescer rapidamente e produzir valores
        imprevisíveis fora do intervalo de interpolação.
    """
    # ordenar coordenadas
    x_ord, y_ord, dy_ord = _ordenar_coordenadas_hermite(
        x_pontos, y_pontos, dy_pontos)

    # criar função de interpolação
    f_interp = _hermite_interp_mat(x_ord, y_ord, dy_ord)

    if f_interp is None:
        print("Erro: Não foi possível criar a função de interpolação.")
        return None
    
    # cálculo do erro (só se f_real for fornecida)
    erro_medio = None
    erro_maximo = None
    if f_real is not None:
        # gera 200 pontos igualmente espaçados no intervalo
        x_test = np.linspace(min(x_ord), max(x_ord), 200)
        y_real = f_real(x_test)
        y_interp = f_interp(x_test)

        erros = np.abs(y_real - y_interp)
        erro_medio = np.mean(erros)
        erro_maximo = np.max(erros)

        print(f"Erro médio absoluto: {erro_medio:.6e}")
        print(f"Erro máximo absoluto: {erro_maximo:.6e}")

    # plot
    # a função de plot também vai mostrar um pouco da extrapolação
    if plot:
        _plotar_hermite(x_ord, y_ord, dy_ord, f_interp, f_real, titulo)

    # a lógica de restrição foi removida.

    return f_interp
