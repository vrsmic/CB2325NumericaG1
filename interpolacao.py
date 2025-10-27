import matplotlib.pyplot as plt
import numpy as np

def lin_interp(x,y):
    def f(x1):
        for i in range (1,len(x)):
            if x[i]>=x1>=x[i-1]:
                a = (y[i]-y[i-1])/(x[i]-x[i-1])
                b = y[i-1]
                
                y1 = b + (x1-x[i-1]) * a
                
                return y1
    return f

def poly_interp(x,y):
    raise NotImplementedError

def hermite_interp(x,y):
    raise NotImplementedError

if __name__=='__main__':
    x = [0, 1, 2, 3]
    y = [1, 2, 0, 4]

    p = lin_interp(x,y)
    print(p(1.5))

    fig, ax = plt.subplots()

    x_points = np.linspace(0, 3, 100)
    y_points = [p(xp) for xp in x_points]

    ax.scatter(x, y)
    ax.plot(x_points, y_points)
    plt.show()