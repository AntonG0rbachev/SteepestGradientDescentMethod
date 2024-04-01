import math

import numpy
from autograd import grad

function = lambda x1, x2: x1 ** 2 + 7 * x2 ** 2 - x1 * x2 + x1
calc_t = lambda x1, x2: ((5 * x1 ** 2 - 32 * x1 * x2 + 4 * x1 + 197 * x2 ** 2 - 2 * x1 + 1) /
                         (26 * x1 ** 2 - 458 * x1 * x2 + 10 * x1 + 2774 * x2 ** 2 - 32 * x2 + 2))
x0 = numpy.array([1.1, 1.1])
eps1 = 0.1
eps2 = 0.15
iterCounts = 10


def norm(args):
    return math.sqrt(sum(map(lambda x: x ** 2, args)))


def steepestGradientDescent(epsilon_1, epsilon_2, max_iter_counts, init_approximation, func, calc_step):
    x_min = numpy.array([])
    gradient = grad(func, [0, 1])
    iter_counts = 0
    check_counts = 0
    new_approximation = init_approximation.copy()
    flag = True
    while flag:
        iter_counts += 1
        gradient_in_point = gradient(new_approximation[0], new_approximation[1])
        grad_norm = norm(gradient_in_point)
        if grad_norm < epsilon_1:
            x_min = new_approximation
            flag = False
            break
        if iter_counts >= max_iter_counts:
            x_min = new_approximation
            flag = False
            break
        t = calc_step(new_approximation[0], new_approximation[1])
        prev_approximation = new_approximation.copy()
        new_approximation = new_approximation - t * numpy.array(gradient_in_point)
        if norm(new_approximation - prev_approximation) < epsilon_2:
            if check_counts == 2:
                x_min = new_approximation
                flag = False
                break
            else:
                check_counts += 1
    return {"x_min": list(x_min), "f(x_min)": func(x_min[0], x_min[1]), "iterations": iter_counts}


print(steepestGradientDescent(eps1, eps2, iterCounts, x0, function, calc_t))
