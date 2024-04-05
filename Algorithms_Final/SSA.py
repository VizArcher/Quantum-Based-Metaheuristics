from random import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj

def SSA(objf, Max_iteration):

    N = 50
    lb = [0.25, 1, 0.3]
    ub = [1, 2, 1]
    dim = 3

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim
    Convergence_curve = np.zeros(Max_iteration)

    # Initialize the positions of salps
    SalpPositions = np.zeros((N, dim))
    for i in range(dim):
        SalpPositions[:, i] = np.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
    SalpFitness = np.full(N, float("inf"))

    FoodPosition = np.zeros(dim)
    FoodFitness = float("inf")

    timerStart = time.time()

    for i in range(0, N):
        SalpFitness[i] = objf(SalpPositions[i, :])

    sorted_salps_fitness = np.sort(SalpFitness)
    I = np.argsort(SalpFitness)

    Sorted_salps = np.copy(SalpPositions[I, :])

    FoodPosition = np.copy(Sorted_salps[0, :])
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 1

    # Main loop
    while Iteration < Max_iteration:

        c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))

        SalpPositions = np.transpose(SalpPositions)  # Move transposition outside the loop

        for i in range(0, N):

            if i < N / 2:
                for j in range(0, dim):
                    c2 = random()
                    c3 = random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * (
                            (ub[j] - lb[j]) * c2 + lb[j]
                        )

            elif i >= N / 2:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2

        SalpPositions = np.transpose(SalpPositions)

        for i in range(0, N):

            # Check if salps go out of the search space and bring them back
            for j in range(dim):
                SalpPositions[i, j] = np.clip(SalpPositions[i, j], lb[j], ub[j])

            SalpFitness[i] = objf(SalpPositions[i, :])

            if SalpFitness[i] > FoodFitness:
                FoodPosition = np.copy(SalpPositions[i, :])
                FoodFitness = SalpFitness[i]

        # Display best fitness along the iteration
        if Iteration % 1 == 0:
            print(str(FoodFitness))

        Convergence_curve[Iteration] = FoodFitness

        Iteration = Iteration + 1

    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    convergence = Convergence_curve
    bestIndividual = FoodPosition

    # Plot the convergence curve
    plt.plot(range(1, Max_iteration+1), Convergence_curve, label='Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function - Cp')
    plt.title('Convergence Curve of SSA')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.show()

    return (bestIndividual, FoodFitness, convergence, executionTime)

if __name__ == '__main__':
    result = SSA(obj, 500)
    print("Best solution:", result[0])
    print("Best fitness:", result[1])
    print("Execution time:", result[3])
