import numpy

function = lambda x1, x2: x1 ** 2 + 7 * x2 ** 2 - x1 * x2 + x1
gradient = lambda x1, x2: numpy.arange([2 * x1 - x2 + 1, 14 * x2 - x1])
t = lambda x1, x2: ((3 * x1 ** 2 - x2 ** 2 + 14 * x1 * x2 + 4 * x1 - 1)
                    / (-16 * x1 ** 3 - 6 * x1 ** 2 + 8 * x2 * x1 ** 2 -
                    2 * x1 * x2 ** 2 + 1358 * x2 ** 2 - 3 * x1 + 14 * x2 - 163 * x1 * x2))
x0 = numpy.arange([1.1, 1.1])
eps1 = 0.1
eps2 = 0.15
iterCount = 10

def steepestGradientDescent():