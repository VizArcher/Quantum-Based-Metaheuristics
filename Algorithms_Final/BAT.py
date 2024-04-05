import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj

def BAT(objf, Max_iteration):

    n = 50
    lb = [0.25, 1, 0.3]
    ub = [1, 2, 1]
    dim = 3
    # Population size

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    N_gen = Max_iteration  # Number of generations

    A = 0.6
    # Loudness  (constant or decreasing)
    r = 0.5
    # Pulse rate (constant or decreasing)

    Qmin = 0  # Frequency minimum
    Qmax = 2  # Frequency maximum

    d = dim  # Number of dimensions

    # Initializing arrays
    Q = np.zeros(n)  # Frequency
    v = np.zeros((n, d))  # Velocities
    Convergence_curve = []

    # Initialize the population/solutions
    Sol = np.zeros((n, d))
    for i in range(dim):
        Sol[:, i] = np.random.rand(n) * (ub[i] - lb[i]) + lb[i]

    S = np.zeros((n, d))
    S = np.copy(Sol)
    Fitness = np.zeros(n)

    # Initialize timer for the experiment
    timerStart = time.time()

    # Evaluate initial random solutions
    for i in range(0, n):
        Fitness[i] = objf(Sol[i, :])

    # Find the initial best solution and minimum fitness
    I = np.argmin(Fitness)
    best = Sol[I, :]
    fmax = max(Fitness)

    # Main loop
    for t in range(0, N_gen):

        # Loop over all bats(solutions)
        for i in range(0, n):
            Q[i] = Qmin + (Qmin - Qmax) * random.random()
            v[i, :] = v[i, :] + (Sol[i, :] - best) * Q[i]
            S[i, :] = Sol[i, :] + v[i, :]

            # Check boundaries
            for j in range(d):
                Sol[i, j] = np.clip(Sol[i, j], lb[j], ub[j])

            # Pulse rate
            if random.random() > r:
                S[i, :] = best + 0.001 * np.random.randn(d)

            # Evaluate new solutions
            Fnew = objf(S[i, :])

            # Update if the solution improves
            if (Fnew >= Fitness[i]) and (random.random() < A):
                Sol[i, :] = np.copy(S[i, :])
                Fitness[i] = Fnew

            # Update the current best solution
            if Fnew >= fmax:
                best = np.copy(S[i, :])
                fmax = Fnew

        # update convergence curve
        Convergence_curve.append(fmax)

        if t % 1 == 0:
            print("At iteration " + str(t) + " the best fitness is " + str(fmax))

    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    convergence = Convergence_curve
    bestIndividual = best

        # Plot the convergence curve
    plt.plot(range(1, Max_iteration+1), Convergence_curve, label='Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function - Cp')
    plt.title('Convergence Curve of BAT')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.show()

    return (bestIndividual, fmax, convergence, executionTime)

if __name__ == '__main__':
    result = BAT(obj, 1000)
    print("Best solution:", result[0])
    print("Best fitness:", result[1])
    print("Execution time:", result[3])
