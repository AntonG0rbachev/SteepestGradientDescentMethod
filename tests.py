from autograd import grad
import numpy

function = lambda x1, x2: (x1 ** 2) + 7 * x2 ** 2 - x1 * x2 + x1
actual_gradient = lambda x1, x2: numpy.array([2 * x1 - x2 + 1, 14 * x2 - x1])
x0 = numpy.array([1.1, 1.1])
gradient = grad(function, (0, 1))
x = x0.copy()
print(actual_gradient(x0[0], x0[1]))
gradient_at_xy = gradient(x0[0], x0[1])
print(gradient_at_xy[0], gradient_at_xy[1])
print(x)
