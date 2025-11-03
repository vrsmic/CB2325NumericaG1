def erro_absoluto(valor_real: float, valor_aprox: float) -> float:
    """Função que calcula o erro absoluto, que corresponde 
    à diferença entre o valor real e o valor aproximado.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro absoluto.
    """
    erro_absoluto = valor_real - valor_aprox
    return abs(erro_absoluto)

def erro_relativo(valor_real: float, valor_aprox: float) -> float:
    '''Função que calcula o erro relativo, que corresponde à diferença entre o 
    valor real e o valor aproximado em comparação com a magnitude do valor real.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro aproximado.
    '''
    if valor_real == 0:
        # Retorna 'infinito' se o valor real é 0, pois o erro é indefinido/infinito
        return float('inf')
    
    erro_relativo = abs(valor_real - valor_aprox)/valor_real
    return abs(erro_relativo)
