import numpy as np
from prettytable import PrettyTable as pt


class Eq:
    def __init__(self, func, y, start, end, eps):
        self.func = func
        self.y = y
        self.start = start
        self.end = end
        self.eps = eps


def f1(x, y):
    return np.cos(x+2) - .3*np.square(y)


def f2(x, y):
    return 1 - np.sin(1.5*x*x+y)


def rk4(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    d = h/6*(k1 + 2*k2 + 2*k3 + k4)
    return d


def adams3(f, x, y, y_prev, y_prev2, h):
    k1 = 23*f(x, y)
    k2 = -16*f(x-h, y_prev)
    k3 = 5*f(x-2*h, y_prev2)
    d = h/12*(k1 + k2 + k3)
    return d


def adams4(f, x, y, y_prev, y_prev2, y_prev3, h):
    k1 = 55*f(x, y)
    k2 = -59*f(x-h, y_prev)
    k3 = 37*f(x-2*h, y_prev2)
    k4 = -9*f(x-3*h, y_prev3)
    d = h/24*(k1 + k2 + k3 + k4)
    return d


def solve_first(f, y, a, b, h, e, method):
    x = a
    i = 0
    array_x = [x]
    array_y = [y]
    if method == 'euler2':
        d_half = 0
        while x < b:
            i += 1
            while 1/3*np.abs(h * f(x, y) - h/2 * f(x, h/2*f(x-h/2, y))) > e:
                h /= 2
            d_half = h * f(x, y)
            y += h * (f(x, y) + f(x+h, y+d_half)) / 2
            x += h
            array_x = np.append(array_x, x)
            array_y = np.append(array_y, y)
    if method == 'rk4':
        while x < b:
            i += 1
            while 1/15*np.abs(rk4(f, x, y, h) - rk4(f, x, rk4(f, x-h/2, y, h/2), h/2)) > e:
                h /= 2
            y += rk4(f, x, y, h)
            x += h
            array_x = np.append(array_x, x)
            array_y = np.append(array_y, y)
    print("Iters: ", i)
    return array_x, array_y


def solve_second(f, init, a, b, h, e, method):
    x = a
    i = 0
    y = init[0]; g = init[1]
    g_prev = 0; y_prev = 0
    g_prev2 = 0; y_prev2 = 0
    array_x = [x]; array_y = [y]
    if method == 'adams3':
        i = 0
        while x < b:
            g_prev2 = g_prev
            g_prev = g
            y_prev2 = y_prev
            y_prev = y
            if i <= 1:
                g += h * rk4(f, x, g, h)
            else:
                while 1/7*np.abs(adams3(f, x, y, y_prev, y_prev2, h) - \
                    adams3(f, x, adams3(f, x, y, y_prev, y_prev2, h/2), y, y_prev, h/2)) > e:
                    h /= 2
                g += adams3(f, x, y, y_prev, y_prev2, h)
            i += 1
            x += h
            if i <= 1:
                y += h * g
            else:
                y += h/12 * (23*g - 16*g_prev + 5*g_prev2)
            array_x = np.append(array_x, x)
            array_y = np.append(array_y, y)
    if method == 'adams4':
        i = 0
        while x < b:
            g_prev3 = g_prev2
            g_prev2 = g_prev
            g_prev = g
            y_prev3 = y_prev2
            y_prev2 = y_prev
            y_prev = y
            if i <= 2:
                g += h * rk4(f, x, g, h)
            else:
                while 1/15*np.abs(adams4(f, x, y, y_prev, y_prev2, y_prev3, h) - \
                    adams4(f, x, adams4(f, x, y, y_prev, y_prev2, y_prev3, h/2), y, y_prev, y_prev2, h/2)) > e:
                    h /= 2
                g += adams4(f, x, y, y_prev, y_prev2, y_prev3, h)
            i += 1
            x += h
            if i <= 2:
                y += h * g
            else:
                y += h/24 * (55*g - 59*g_prev + 37*g_prev2 - 9*g_prev3)
            array_x = np.append(array_x, x)
            array_y = np.append(array_y, y)
    print("Iters: ", i)
    return array_x, array_y



eq = Eq(f1, 0, .0, .5, .001)
answer_x, answer_y = solve_first(eq.func, eq.y, eq.start, eq.end, .1, eq.eps, 'euler2')

# eq = Eq(f2, [0, 1], .0, .5, .001)
# answer_x, answer_y = solve_second(eq.func, eq.y, eq.start, eq.end, .1, eq.eps, 'adams4')

table = pt()
table.add_column("x", answer_x)
table.add_column("y", answer_y)
# table.add_column("y'", answer_g)
print(table[::]) # Показать таблицу с начало:конец:каждый
