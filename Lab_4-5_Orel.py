"""
    Автор: Орел Максим
    Группа: КБ-161
    Вариант: 11
    Дата создания: 27/03/2018
    Python Version: 3.6
"""

import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Constants
accuracy = 0.00001
START_X = -10
END_X = 10
START_Y = -10
END_Y = 10


# неявные функции
def f1(x, y):
    return 2 * y + np.cos(x) - 2


def f2(x, y):
    return np.sin(y + 1) - x - 1.2


# явные функции
def _f1(x):
    return 1 - np.cos(x) / 2


def _f2(y):
    return np.sin(y + 1) - 1.2


# 0 - means derivative with respect to x, 1 - means derivative with respect to y
def diff_f(func, var=0, point=None):
    if point is None:
        point = []
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)

    return derivative(wraps, point[var], dx=1e-6)


# Строит графики функций
def build_function_xy(func, legend="", colors="black", di=0.01, x_from=START_X, x_to=END_X, y_from=START_Y,
                      y_to=END_Y):
    plt.axis([START_X, END_X, START_Y, END_Y])
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    x_array = np.arange(x_from, x_to, di)
    y_array = np.arange(y_from, y_to, di)

    x, y = np.meshgrid(x_array, y_array)
    f = func(x, y)

    cs = plt.contour(x, y, f, [0], colors=colors)

    # plt.clabel(cs, inline=1, fontsize=10)     # this can make label on the lines
    plt.title('Simplest graphs with labels')

    cs.collections[0].set_label(legend)
    plt.legend(loc="upper right")


# Отрисовывает функции
def show_plot():
    plt.figure()
    plt.show()


def print_result(x, y, f1, f2, n):
    temp_x_e = str("%.2e" % x)
    temp_x_f = str("%.6f" % x)
    temp_y_e = str("%.2e" % y)
    temp_y_f = str("%.6f" % y)

    temp_iterations = str(n)

    temp_f1_x_e = str("%.2e" % f1)
    temp_f1_x_f = str("%.6f" % f1)
    temp_f2_x_e = str("%.2e" % f2)
    temp_f2_x_f = str("%.6f" % f2)

    print(
        ("Корень системы нелинейных уравнений: x = {0} ({1}), y = {2} ({3})\n" +
         "Найдены за {4} итерации(ий) \n" +
         "Функции в этих точках: f1(x,y) = {5} ({6}) <= {9}  (точность); f2(x,y) = {7} ({8}) <= {9} (точность)").format(
            temp_x_e, temp_x_f, temp_y_e, temp_y_f,
            temp_iterations,
            temp_f1_x_e, temp_f1_x_f, temp_f2_x_e, temp_f2_x_f,
            accuracy
        )
    )


def iteration_space_station(x0, y0, __f1, __f2, n):
    if abs(f1(x0, y0)) < accuracy and abs(f2(x0, y0)) < accuracy:
        print("_______________________________________________________________________________________________________")
        print("Метод итерации:")
        print_result(x0, y0, f1(x0, y0), f2(x0, y0), n)
    else:
        y = __f1(x0)
        x = __f2(y0)

        return iteration_space_station(x, y, __f1, __f2, n + 1)


def newton_piston(x0, y0, _f1, _f2, n):
    if abs(f1(x0, y0)) < accuracy and abs(f2(x0, y0)) < accuracy:
        print("_______________________________________________________________________________________________________")
        print("Метод Ньютона:")
        print_result(x0, y0, f1(x0, y0), f2(x0, y0), n)
    else:

        detJ = diff_f(f1, 0, [x0, y0]) * diff_f(f2, 1, [x0, y0]) - diff_f(f1, 1, [x0, y0]) * diff_f(f2, 0, [x0, y0])
        detA1 = f1(x0, y0) * diff_f(f2, 1, [x0, y0]) - diff_f(f1, 1, [x0, y0]) * f2(x0, y0)
        detA2 = diff_f(f1, 0, [x0, y0]) * f2(x0, y0) - f1(x0, y0) * diff_f(f2, 0, [x0, y0])

        x = x0 - detA1 / detJ
        y = y0 - detA2 / detJ

        return newton_piston(x, y, _f1, _f2, n + 1)


if __name__ == "__main__":
    # Отключение вывода некоторых уведомлений
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    # Устанавливает максимальную глубину рекурсии на 2000
    sys.setrecursionlimit(2000)

    # нарисуем функцию и её производной на промежутке заданном впараметрах проагрммы сверху
    print("Выведен график функции и график производной")
    build_function_xy(f1, "функция f1(x)", "green")
    build_function_xy(f2, "функция f2(x)", "blue")
    show_plot()

    try:
        iteration_space_station(0, 0, _f1, _f2, 0)
    except Exception as e:
        print(e)

    print()

    try:
        newton_piston(-1, -1, f1, f2, 0)
    except Exception as e:
        print(e)
