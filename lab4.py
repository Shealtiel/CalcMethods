import numpy as np
from prettytable import PrettyTable as pt


def lagr(x, y, t):
    z = 0
    # d = 0
    for j in range(len(y)):
        p1 = 1
        p2 = 1
        for i in range(len(x)):
            if i != j:
                p1 *= (t - x[i])
                p2 *= (x[j] - x[i])
        z += y[j] * p1 / p2
        # d += y[j] / p2
    return z


def eitken(x, y, t):
    n = len(x)
    e = np.zeros((n, n))
    e[0, :] = y
    for j in range(0, n - 1):
        for i in range(j + 1, n):
            e[j + 1, i] = (e[j, i] * (t - x[j]) - e[j, j] * (t - x[i])) / (x[i] - x[j])
    return e


def func_part2(x):
    return np.log(x)


def left_der_1(x, h):
    return (func_part2(x) - func_part2(x - h)) / h


def right_der_1(x, h):
    return (func_part2(x + h) - func_part2(x)) / h


def der_1(x, h):
    return (func_part2(x + h) - func_part2(x - h)) / (2 * h)


def der_2(x, h):
    return (func_part2(x + h) + func_part2(x - h) - 2 * func_part2(x)) / (h * h)


def error(exact, approx):
    return np.abs((1 - approx / exact))


def main():
    x0 = 1.4
    y0 = x0 + 10 / x0
    x1 = np.array([1., 1.5, 2., 2.5])
    y1 = np.array([11., 8.167, 7., 6.5])
    x_star = np.array([1.036])
    x2 = np.array([1., 1.08, 1.2, 1.27, 1.31, 1.38])
    y2 = np.array([1.1752, 1.30254, 1.50946, 1.2173, 1.22361, 1.2347])

    h0 = .1
    ans = lagr(x1, y1, x0)
    der = (lagr(x1, y1, x0 + h0) - lagr(x1, y1, x0 - h0)) / (2 * h0)
    err = error(y0, ans)
    print("Ответ=%f Ошибка=%f Производная~%f" % (ans, err, der))

    et = eitken(x2, y2, x_star)
    print("Схема Эйткена:\n", et.T)

    a3 = 3
    b3 = 6
    m3 = 3.04
    n = 5
    h = (b3 - a3) / (n-1)
    x3 = np.sort(np.append(np.linspace(a3, b3, num=n), m3))

    ader1 = 1/x3
    ader2 = -1/np.square(x3)

    lder = []
    rder = []
    fder = []
    sder = []


    for i in x3:
        lder.append(left_der_1(i, h))
        rder.append(right_der_1(i, h))
        fder.append(der_1(i, h))
        sder.append(der_2(i, h))

    lerr = error(ader1, lder)
    rerr = error(ader1, rder)
    ferr = error(ader1, fder)
    serr = error(ader2, sder)

    table = pt()
    table.add_column("x", x3, align="l")
    # table.add_column("y", y3)
    table.add_column("1 производная левая", lder, align="l")
    table.add_column("Ошибка", lerr, align="l")
    table.add_column("1 производная правая", rder, align="l")
    table.add_column("Ошибка", rerr, align="l")
    table.add_column("1 производная 2 порядка", fder, align="l")
    table.add_column("Ошибка", ferr, align="l")
    table.add_column("Аналит. 1 производная", ader1, align="l")
    table.add_column("2 производная", sder, align="l")
    table.add_column("Ошибка", serr, align="l")
    table.add_column("Аналит. 2 производная", ader2, align="l")
    print(table)
    print("Макс ошибка 1 левой производной = %f" % np.max(lerr))
    print("Макс ошибка 1 правой производной = %f" % np.max(rerr))
    print("Макс ошибка 1 производной вт. пор. = %f" % np.max(ferr))
    print("Макс ошибка 2 производной = %f" % np.max(serr))

main()
