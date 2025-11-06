def erro_absoluto(valor_real: float, valor_aprox: float) -> float:
    """Função que calcula o erro absoluto, que corresponde 
    à diferença entre o valor real e o valor aproximado.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro absoluto.
    """
    return abs(valor_real - valor_aprox)

def erro_relativo(valor_real: float, valor_aprox: float) -> float:
    """Função que calcula o erro relativo, que corresponde à diferença entre o 
    valor real e o valor aproximado em comparação com a magnitude do valor real.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro relativo.

    Raises:
        ZeroDivisionError: Se 'valor_real' for zero, pois a divisão seria indefinida.
    """
    if valor_real == 0.0:
        raise ZeroDivisionError("Não é possível calcular o erro relativo quando o 'valor_real' é zero (divisão por zero).")
        
    erro_abs = erro_absoluto(valor_real, valor_aprox)
    return abs(erro_abs/valor_real)
