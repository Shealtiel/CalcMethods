import numpy as np


def rearrange(array, j):  # Перемещает строку с максимальным элементом в выбранном столбце наверх
    maxind = np.argmax(np.abs(array[:, j]))  # Индекс максимального элемента
    sort_ind = np.arange(0, array.shape[0])  # Все индексы
    sort_ind[0], sort_ind[maxind] = sort_ind[maxind], sort_ind[0]  # Своп элементов
    array = array[sort_ind, :]
    return array


def divide_substract(array, i, j):
    array[i + 1, :] -= array[0, :] * (array[i + 1, j] / array[0, j])
    return array


def find_roots(array):
    n = array.shape[0]
    roots = np.zeros(n)
    for i in range(n - 1, -1, -1):
        roots[i] = array[i][n] / array[i][i]
        for j in range(i - 1, -1, -1):
            array[j][n] -= array[j][i] * roots[i]
    return roots


def gauss(array, b):
    array = np.append(array, b, axis=1)
    n = array.shape[0]
    # print(n)
    array_result = np.zeros(array.shape)
    array_result[0] += array[0]
    for j in range(n - 1):
        array = rearrange(array, j)
        for i in range(n - j - 1):
            array = divide_substract(array, i, j)
        array = np.delete(array, 0, 0)
        array_result[j+1] += array[0]
    return array_result


def error(exact, approx):
    return np.abs((exact - approx) / approx)


def seidel(A, b, init):
    Cm = np.dot(A.T, A)
    dm = np.dot(A.T, b)

    lower = np.tril(Cm, -1)      # нижняя часть
    upper = np.triu(Cm, 1)   # верхняя часть
    diag = np.diag(np.diag(Cm))  # диагональные элементы
    ldi = np.linalg.inv(lower+diag)

    t = -np.dot(ldi, upper)
    c = np.dot(ldi, dm)

    while True:
        dk = np.dot(t, init)
        old_init = init
        init = dk + c
        if np.linalg.norm(init - old_init) < 1e-7:
            return init


A = np.array([[1.15, 0.42, 10.10, 4.25],
              [1.59, 0.55, -0.32, 0.29],
              [1.14, 3.15, 2.05, 7.86],
              [0.77, 6.11, -3.01, 0.74]])
b = np.array([[15.08], [1.01], [7.90], [-7.61]])
x_exact = np.array([1, -1, 1, 1])
x0 = np.array([[.5], [-0.5], [.5], [.5]])

x_approx_1 = find_roots(gauss(A, b))
x_approx_2 = seidel(A, b, x0)

print(f'Gauss\'s method \n{x_approx_1}')
print(f'Zeidel\'s method \n{x_approx_2}')

# print(error(x_exact, x_approx_2))
