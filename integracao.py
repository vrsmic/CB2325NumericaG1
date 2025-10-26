import matplotlib.pyplot as plt
import math as mt
import random
import numpy as np

def function_one_variable(x):
    f = mt.sin(x)
    return f

def function_two_variables(x,y):
    f = (x**2)*y
    return f


def monte_carlo(a,b, function, n):
    inside_x, inside_y = [], []
    outside_x, outside_y = [], []

    for _ in range(n):
        point_x = random.uniform(a, b)
        point_y = random.uniform(0, 1)
        value_of_f = function(point_x)
        
        if point_y <= value_of_f:
            inside_x.append(point_x)
            inside_y.append(point_y)
        else:
            outside_x.append(point_x)
            outside_y.append(point_y)
    
    area = abs(b-a)*len(inside_x)/n
    xs = np.linspace(a, b, 400)
    ys = [function(x) for x in xs]
    plt.plot(xs, ys, 'k-', label='f(x) = sin(x)')
    
    plt.scatter(inside_x, inside_y, color='green', s=10, label='Dentro')
    plt.scatter(outside_x, outside_y, color='red', s=10, label='Fora')

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
integral2 = monte_carlo_two_variables(0,1,0,1,function_two_variables, n)
integral = monte_carlo(0,mt.pi,function_one_variable,n)
print(f'Com {n} pontos, o valor calculado da integral[0,1]x[0,1] de f(x,y) = x²y foi {integral2}\n'
      f'e o valor da integral[0,pi] de f(x) = sin(x) foi {integral}')

