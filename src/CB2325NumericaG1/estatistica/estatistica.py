import numpy as np
import numpy.typing as npt

def mean(x: npt.ArrayLike, pesos: npt.ArrayLike = None) -> float:
    """
    Calcula a média aritmética simples de um conjunto de dados, caso apenas um paramêtro for passado
    Se um segundo parâmetro for passado, esse será considerado um conjunto de pesos, e será
    calculada a média aritmética ponderada.

    Args:
        x (npt.ArrayLike): 
            Vetor com o conjunto de dados para calcular a média
        pesos (npt.ArrayLike, opcional): 
            Vetor contendo os valores da variável dependente.

    Returns:
        float: 
            Média simples ou ponderada dos dados, de acordo com os parâmetros passados    
        
    Raises:
        ValueError: 
            Se 'x' e 'pesos' tiverem comprimentos diferentes.
            Se a soma dos valores de 'pesos' for igual à 0
    """
    
    if pesos == None:
        pesos = np.ones(len(x))
    
    if len(pesos) != len(x):
        raise ValueError("Os arrays 'x' e 'pesos' devem ter o mesmo comprimento")

    sumx = 0.0
    sumpesos = 0.0
    for v, w in zip(x, pesos):
        sumx += v*w
        sumpesos += w

    media = sumx/sumpesos

    return float(media)

def std(x: npt.ArrayLike, pesos: npt.ArrayLike = None) -> float:
    """
    Calcula o desvio padrão de um conjunto de dados

    Args:
        x (npt.ArrayLike): 
            Vetor com o conjunto de dados para calcular o desvio padrão

    Returns:
        float: 
            Desvio padrão calculado 
    """
    
    media = mean(x)
    variancia = 0.0
    for v in x:
        variancia += (v-media)**2
    
    variancia /= len(x)

    return float(variancia**0.5)
