import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt


def func(x):
    return np.power(x, 4) - np.sqrt(x + 1) - 3


def derivative_func(x):
    return 4*np.power(x, 3) - 1/(2*np.sqrt(x + 1))


def newton(init):
    while True:
        d = func(init)/derivative_func(init)
        init -= d
        print(init)
        if np.abs(d) < eps:
            return init


# Обьявления
eps = 1e-7
a = 1
b = 2
n = 20
x = np.linspace(a, b, n)  # шаг (b - a)/n
y = func(x)
der_y = derivative_func(x)

# Метод Ньютона
newton(b)

# Табличка
table_1 = pt()
table_1.add_column("x", x)
table_1.add_column("f(x)", y)
table_1.add_column("f'(x)", der_y)
print(table_1)

# График
plt.plot(x, y)
plt.grid(True)
plt.show()
