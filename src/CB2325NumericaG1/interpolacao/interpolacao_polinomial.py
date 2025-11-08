import numpy as np

def poly_interp(x_val: list,
                y_val: list,
                plot: bool = False):

    n = len(x_val)

    def P(x):
        acc_som = 0
        # loop do somatório
        for j in range(n):
            # acumulador
            acc_prod = 1
            
            # o produtorio exije que j != i
            x_val_sj = np.delete(x_val, j)

            # loop do produtório
            for i in range(n):
                if j == i: 
                    continue
                acc_prod *= (x - x_val[i])/(x_val[j] - x_val[i])
            
            acc_som += y_val[j] * acc_prod
        
        resultado = acc_som

        return resultado
    
    return P


# testes

pol = poly_interp([1, 3, 5], [2, 4, 6])

print(pol(10))


# receber pontos (quantos eu quiser?)
# achar polinomio que passe por todos esses pontos (metodo de lagrange)
# retornar expressao desse polinomio
# representar visualmente o polinomio e os pontos dados (talvez colocar um parametro a mais na funcao pra ver se plota ou nao)


# existe todo um metodozinho do lagrange pra calcular po
# vou seguir ele da maneira mais simples, depois vejo como vai ficar a complexidade
# nah, tenho que fazer com numpy, tinha esquecido