from decimal import Decimal

def erro_absoluto(valor_real,valor_aprox):
    """Função que calcula o erro absoluto, que corresponde 
    à diferença entre o valor real e o valor aproximado.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro absoluto.
    """
    erro_absoluto=valor_real-valor_aprox
    return abs(erro_absoluto)

def erro_relativo(valor_real,valor_aprox):
    '''Função que calcula o erro relativo, que corresponde 
    à diferença entre o valor real e o valor aproximado em 
    comparação com a magnitude do valor real.
    
    Args:
    valor_real: valor exato
    valor_aprox: valor aproximado

    Returns:
    Retorna o valor do erro aproximado.
    '''
    erro_relativo=abs(valor_real-valor_aprox)/valor_real
    return abs(erro_relativo)


valor_real=Decimal(input())
valor_aprox=Decimal(input())
print(erro_absoluto(valor_real, valor_aprox))
print(erro_relativo(valor_real, valor_aprox))