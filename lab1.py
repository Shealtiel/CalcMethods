import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable as pt


def func(x):
    return np.power(x, 4) - np.sqrt(x + 1) - 3


def derivative_func(x):
    return 4*np.power(x, 3) - 1/(2*np.sqrt(x + 1))


# def error(x, m):
#     return np.abs(func(x))/m


def newton(init):
    while True:
        d = func(init)/derivative_func(init)
        init -= d
        # print(init)
        if np.abs(d) < eps:
            return init


def hordes(init, a, b):
    while True:
        if init == a:
            d = func(init)*(b - init)/(func(b) - func(init))
        else:
            d = func(init)*(init - a)/(func(init) - func(a))
        init -= d
        # print(init)
        if np.abs(d) < eps:
            return init


def secant(init0, init1):
    while True:
        d = func(init1)*(init1 - init0)/(func(init1) - func(init0))
        init1 -= d
        init0 += d
        # print(init1)
        if np.abs(d) < eps/10:
            return init1


def finite_dif(init):
    h = 5e-2
    while True:
        d = h*func(init)/(func(init+h)-func(init))
        init -= d
        # print(init)
        if np.abs(d) < eps:
            return init


def steffensen(init):
    while True:
        d = (func(init)**2)/(func(init + func(init)) - func(init))
        init -= d
        # print(init)
        if np.abs(d) < eps:
            return init


def simple(init):
    while True:
        # print(init)
        d = func(init)*0.1
        init -= d
        if np.abs(d) < eps:
            return init


# Обьявления
eps = 1e-7
a = 1
b = 2
n = 20
x0 = a if func(a)*derivative_func(a) > 0 else b
x = np.linspace(a, b, n)  # шаг (b - a)/n
y = func(x)
der_y = derivative_func(x)
m = np.amin(np.abs(der_y))

print("Метод Ньютона\n", newton(x0))
print("Метод хорд\n", hordes(x0, a, b))
print("Метод секущих\n", secant(a, b))
print("Конечноразностный метод\n", finite_dif(x0))
print("Метод Стеффенсена\n", steffensen(x0))
print("Метод простых итераций\n", simple(x0))


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