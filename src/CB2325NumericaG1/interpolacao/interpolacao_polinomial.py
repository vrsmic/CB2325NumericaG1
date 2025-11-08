import numpy as np

def poly_interp(x_val: list,
                y_val: list,
                plot: bool = False):

    n = len(x_val)

    x_val_np = np.array(x_val)
    y_val_np = np.array(y_val)

    def P(x):
        acc_som = 0

        # loop do somatório
        for j in range(n):          
            # o produtorio exije que j != i
            x_val_sj = np.delete(x_val_np, j)

            # calculando numeradores e denominadores do produtorio
            num = x - x_val_sj
            den = x_val_np[j] - x_val_sj

            acc_som += y_val_np[j] * np.prod(num / den)
        
        return acc_som
    
    return P

# testes

pol = poly_interp([1, 3, 5], [2, 4, 6])

print(pol(87))


# tenho que comentar e melhorar o nome das variaveis; codigo precisa estar legivel