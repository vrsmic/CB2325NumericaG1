import matplotlib.pyplot as plt
import math as mt
import random
import numpy as np


function_of_two_variables = lambda x, y : (x**2)*y


function_of_one_variable = lambda x: x**3



def monte_carlo(a,b, function, n):
    values = 0
    
    for _ in range(n):
        point_x = random.uniform(a, b)
        value_of_f = function(point_x)
        values += value_of_f

    avarage_value_of_f = values/n
    area = abs(b-a)*avarage_value_of_f
    xs = np.linspace(a, b, 400)
    ys = [function(x) for x in xs]
    plt.plot(xs, ys, 'k-', label='f(x)')
    
    vetor_para_plot_de_f = [avarage_value_of_f for x in xs]
    plt.plot(xs, vetor_para_plot_de_f, 'k-', label='Avarage Value of f')
    plt.fill_between(xs,vetor_para_plot_de_f,color='blue',alpha=0.5, label='equivalent area')
    plt.fill_between(xs,ys,color='green',alpha=0.5,label='Area under the curve')

    plt.title(f"Método de Monte Carlo\nAproximação da área ≈ {area:.4f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    
    return area


def monte_carlo_two_variables(a,b,c,d,function, n):
    values = 0
    for i in range(n):
        point_x = random.uniform(a,b)
        point_y = random.uniform(c,d)
        value_of_f = function(point_x,point_y)
        values += value_of_f
    
    avarage_ceiling_of_f = values/n
    area_of_domain = abs(b-a)*abs(d-c)
    volume = avarage_ceiling_of_f*area_of_domain
    return volume

n = int(input())
integral2 = monte_carlo_two_variables(0,1,0,1,function_of_two_variables, n)
integral = monte_carlo(0,mt.pi,function_of_one_variable,n)
print(f'Com {n} pontos, o valor calculado da integral[0,1]x[0,1] de f(x,y) = x²y foi {integral2}\n'
      f'e o valor da integral[0,pi] de f(x) = sin(x) foi {integral}')

