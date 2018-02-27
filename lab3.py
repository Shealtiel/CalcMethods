import numpy as np
from prettytable import PrettyTable as pt


def integral_function(x):
    return (np.exp(x) - 1) / (np.exp(x) + 1)


def primitive(x):
    return 2 * np.log(np.exp(x) + 1) - x


def left_rectangles(y):
    y = np.delete(y, len(y) - 1)
    return np.sum(y)


def right_rectangles(y):
    y = np.delete(y, 0)
    return np.sum(y)


def middle_rectangles(y):
    y = np.delete(y, 0)
    return np.sum(y)


def trapeze(y):
    y1_n = (y[0] + y[len(y) - 1]) / 2
    y = np.delete(y, [0, len(y) - 1])
    return y1_n + np.sum(y)


def simpson(y):
    y_0 = np.delete(y, [0, len(y) - 1])
    y_1 = np.delete(y, 0)
    return (y[0] + 2 * np.sum(y_0[::2]) + 4 * np.sum(y_1[1::2]) + y[0] + y[len(y) - 1])/3


def runge(a, b, n, step, eps):
    h = (b - a) / n
    x = np.linspace(a, b, n) - h * step
    integral = integral_function(x)
    while np.max(np.subtract(integral[1::2], integral[::2])) >= eps / 10:
        n *= 2
        h /= 2
        x = np.linspace(a, b, n) - h * step
        integral = integral_function(x)
    return h * integral, n


def error(exact, approx):
    return np.abs((1 - approx / exact)) * 100


def main():
    eps = 1e-4
    a = 0
    b = 3
    n = 10
    exact = primitive(b) - primitive(a)

    runge_0 = runge(a, b, n, 0, eps)
    integral_0 = runge_0[0]
    n_0 = runge_0[1]
    h_0 = (b - a) / n

    runge_1 = runge(a, b, n, 1/2, eps)
    integral_1 = runge_1[0]
    n_1 = runge_1[1]
    h_1 = (b - a) / n

    #  Ответы
    lr_ans = left_rectangles(integral_0)
    lr_err = error(exact, lr_ans)

    rr_ans = right_rectangles(integral_0)
    rr_err = error(exact, rr_ans)

    mr_ans = middle_rectangles(integral_1)
    mr_err = error(exact, mr_ans)

    tr_ans = trapeze(integral_0)
    tr_err = error(exact, tr_ans)

    sm_ans = simpson(integral_0)
    sm_err = error(exact, sm_ans)

    # Табличка
    table_1 = pt()
    table_1.add_column("Метод", ["Точный", "Лев.кв", "Прав.кв", "Сред.кв", "Трапеций", "Симпсона"])
    table_1.add_column("Интеграл", [exact, lr_ans, rr_ans, mr_ans, tr_ans, sm_ans])
    table_1.add_column("Шаг", ["-", h_0, h_0, h_0, h_1, h_0])
    table_1.add_column("К-во ш-в", ["-", n_0, n_0, n_0, n_1, n_0])
    table_1.add_column("Погрешность", ["-", lr_err, rr_err, mr_err, tr_err, sm_err])
    print(table_1)


main()
