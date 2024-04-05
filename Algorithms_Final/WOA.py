from random import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj

def WOA(objf, Max_iter):

    SearchAgents_no = 50
    lb = [0.25, 1, 0.3]
    ub = [1, 2, 1]
    dim = 3

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    # initialize position vector and score for the leader
    Leader_pos = np.zeros(dim)
    Leader_score = float("-inf")  # change this to inf for minimization problems

    # Initialize the positions of search agents
    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            np.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    timerStart = time.time()
    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = np.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

            # Update the leader
            if fitness > Leader_score:  # Change this to < for minimization problem
                Leader_score = fitness
                # Update alpha
                Leader_pos = np.copy(Positions[i, :])

        a = 2 - t * ((2) / Max_iter)
        # a decreases linearly from 2 to 0 in Eq. (2.3)

        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / Max_iter)

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            r1 = random()  # r1 is a random number in [0,1]
            r2 = random()  # r2 is a random number in [0,1]

            A = 2 * a * r1 - a  # Eq. (2.3) in the paper
            C = 2 * r2  # Eq. (2.4) in the paper

            b = 1
            # parameters in Eq. (2.5)
            l = (a2 - 1) * random() + 1  # parameters in Eq. (2.5)

            p = random()  # p in Eq. (2.6)

            for j in range(0, dim):

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(SearchAgents_no * random())
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand

                    elif abs(A) < 1:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader

                elif p >= 0.5:
                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    # Eq. (2.5)
                    Positions[i, j] = (
                        distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                        + Leader_pos[j]
                    )

        convergence_curve[t] = Leader_score
        if t % 1 == 0:
            print(f"At iteration {t + 1}, the best fitness is {Leader_score}")
        t += 1

    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    convergence = convergence_curve
    bestIndividual = Leader_pos

    # Plot the convergence curve
    plt.plot(range(Max_iter), convergence_curve, label='Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function - Cp')
    plt.title('Convergence Curve of WOA')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.show()

    return (bestIndividual, Leader_score, convergence, executionTime)

if __name__ == '__main__':
    result = WOA(obj, 1000)
    print("Best solution:", result[0])
    print("Best fitness:", result[1])
    print("Execution time:", result[3])
