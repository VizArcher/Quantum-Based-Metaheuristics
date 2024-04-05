import math
import numpy
import random
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj

def get_cuckoos(nest, best, lb, ub, n, dim):

    # perform Levy flights
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.array(nest)
    beta = 3 / 2
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    s = numpy.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = numpy.random.randn(len(s)) * sigma
        v = numpy.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * numpy.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = numpy.clip(s[k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf):
    # Evaluating all new solutions
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.copy(nest)

    for j in range(0, n):
        # for j=1:size(nest,1),
        fnew = objf(newnest[j, :])
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    # Find the current best

    fmax = max(fitness)
    K = numpy.argmax(fitness)
    bestlocal = tempnest[K, :]

    return fmax, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim):

    # Discovered or not
    tempnest = numpy.zeros((n, dim))

    K = numpy.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (
        nest[numpy.random.permutation(n), :] - nest[numpy.random.permutation(n), :]
    )

    tempnest = nest + stepsize * K

    return tempnest


def CS(objf, N_IterTotal):

    n = 50
    lb = [0.25, 1, 0.3]
    ub = [1, 2, 1]
    dim = 3

    # Discovery rate of alien eggs/solutions
    pa = 0.4

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    
    # Initialize convergence
    convergence_curve = []

    # RInitialize nests randomely
    nest = numpy.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    new_nest = numpy.zeros((n, dim))
    new_nest = numpy.copy(nest)

    bestnest = [0] * dim

    fitness = numpy.zeros(n)
    fitness.fill(float("inf"))

    print('CS is optimizing  "' + objf.__name__ + '"')
    timerStart = time.time()

    fmax, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

    t = 0
    # Main loop counter
    while t < N_IterTotal:
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        new_nest = empty_nests(new_nest, pa, n, dim)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf)

        if fnew > fmax:
            fmax = fnew
            bestnest = best

        convergence_curve.append(fmax)

        if t % 1 == 0:
            print(["At iteration " + str(t) + " the best fitness is " + str(fmax)])
        t += 1

    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    convergence = convergence_curve
    bestIndividual = bestnest

    # Plot the convergence curve
    plt.plot(range(1,N_IterTotal+1), convergence_curve, label='Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function - Cp')
    plt.title('Convergence Curve of CSO')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.show()

    return (bestIndividual, fmax, convergence, executionTime)

if __name__ == '__main__':
    result = CS(obj, 101)
    print("Best solution:", result[0])
    print("Best fitness:", result[1])
    print("Execution time:", result[3])