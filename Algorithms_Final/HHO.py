from random import random
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj


def HHO(objf, Max_iter):

    SearchAgents_no = 50
    lb = [0.25, 1, 0.3]
    ub = [1, 2, 1]
    dim = 3

    # initialize the location and Energy of the rabbit
    Rabbit_Location = np.zeros(dim)
    Rabbit_Energy = float("-inf")  # change this to inf for minimization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = np.asarray(lb)
    ub = np.asarray(ub)

    # Initialize convergence
    convergence_curve = np.zeros(Max_iter)

    # Initialize the locations of Harris' hawks
    X = np.asarray(
        [x * (ub - lb) + lb for x in np.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    timerStart = time.time()

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = np.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :])

            # Update the location of Rabbit
            if fitness > Rabbit_Energy:  # Change this to < for minimization problem
                Rabbit_Energy = fitness
                Rabbit_Location = X[i, :].copy()

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):

            E0 = 2 * random() - 1  # -1<E0<1
            Escaping_Energy = E1 * (
                E0
            )  # escaping energy of rabbit Eq. (3) in the paper

            # -------- Exploration phase Eq. (1) in paper -------------------

            if abs(Escaping_Energy) >= 1:
                # Harris' hawks perch randomly based on 2 strategy:
                q = random()
                rand_Hawk_index = math.floor(SearchAgents_no * random())
                X_rand = X[rand_Hawk_index, :]
                if q < 0.5:
                    # perch based on other family members
                    X[i, :] = X_rand - random() * abs(
                        X_rand - 2 * random() * X[i, :]
                    )

                elif q >= 0.5:
                    # perch on a random tall tree (random site inside group's home range)
                    X[i, :] = (Rabbit_Location - X.mean(0)) - random() * (
                        (ub - lb) * random() + lb
                    )

            # -------- Exploitation phase -------------------
            elif abs(Escaping_Energy) < 1:
                # Attacking the rabbit using 4 strategies regarding the behavior of the rabbit

                # phase 1: ----- surprise pounce (seven kills) ----------
                # surprise pounce (seven kills): multiple, short rapid dives by different hawks

                r = random()  # probablity of each event

                if (
                    r >= 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (6) in paper
                    X[i, :] = (Rabbit_Location) - Escaping_Energy * abs(
                        Rabbit_Location - X[i, :]
                    )

                if (
                    r >= 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (4) in paper
                    Jump_strength = 2 * (
                        1 - random()
                    )  # random jump strength of the rabbit
                    X[i, :] = (Rabbit_Location - X[i, :]) - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )

                # phase 2: --------performing team rapid dives (leapfrog movements)----------

                if (
                    r < 0.5 and abs(Escaping_Energy) >= 0.5
                ):  # Soft besiege Eq. (10) in paper
                    # rabbit try to escape by many zigzag deceptive motions
                    Jump_strength = 2 * (1 - random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X[i, :]
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # hawks perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X[i, :])
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()
                if (
                    r < 0.5 and abs(Escaping_Energy) < 0.5
                ):  # Hard besiege Eq. (11) in paper
                    Jump_strength = 2 * (1 - random())
                    X1 = Rabbit_Location - Escaping_Energy * abs(
                        Jump_strength * Rabbit_Location - X.mean(0)
                    )
                    X1 = np.clip(X1, lb, ub)

                    if objf(X1) < fitness:  # improved move?
                        X[i, :] = X1.copy()
                    else:  # Perform levy-based short rapid dives around the rabbit
                        X2 = (
                            Rabbit_Location
                            - Escaping_Energy
                            * abs(Jump_strength * Rabbit_Location - X.mean(0))
                            + np.multiply(np.random.randn(dim), Levy(dim))
                        )
                        X2 = np.clip(X2, lb, ub)
                        if objf(X2) < fitness:
                            X[i, :] = X2.copy()

        convergence_curve[t] = Rabbit_Energy
        if t % 1 == 0:
            print( "At iteration " + str(t) + " the best fitness is " + str(Rabbit_Energy))
        
        t = t + 1

    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    convergence = convergence_curve
    bestIndividual = Rabbit_Location

    # Plot the convergence curve
    plt.plot(range(1, Max_iter+1), convergence_curve, label='Convergence Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function - Cp')
    plt.title('Convergence Curve of HHO')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    plt.legend()
    plt.show()

    return (bestIndividual, Rabbit_Energy, convergence, executionTime)

def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    zz = np.power(np.absolute(v), (1 / beta))
    step = np.divide(u, zz)
    return step

if __name__ == '__main__':
    result = HHO(obj, 500)
    print("Best solution:", result[0])
    print("Best fitness:", result[1])
    print("Execution time:", result[3])
