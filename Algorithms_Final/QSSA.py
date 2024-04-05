import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from Kriging_Surrogate_Model import obj

pop_size = 50
iterations = 300
lower = [0.25, 1, 0.3]
upper = [1, 2, 1]
dim = 3

class QSSA():
    def __init__(self, pop_size=pop_size, dim=dim, iterations=iterations, lower=lower, upper=upper):
        self.pop_size = pop_size
        self.iterations = iterations

        self.X = np.zeros((pop_size, dim))  # Position
        self.Xs = np.zeros((pop_size, dim))
        self.Xe = np.zeros((pop_size, dim))
        self.Xm = np.zeros((pop_size, dim))  # Solutions
        self.dim = dim  # Dimension

        self.Fmax = 0
        self.LB = [0]*self.dim
        self.UB = [0]*self.dim
        self.Fitness = [0]*self.pop_size
        self.pbest = [0]*self.pop_size
        self.gbest = 0

        # Hyperparameters
        self.g1 = 250
        self.epsilon = 0.1

        #(g1,epsilon) :- (0.01, 500) , (0.1, 250) , (1 , 100) , (50, 50) , (100, 1) , (250,  0.1) , (500, 0.01) 

    def best_salp(self):
        i = 0
        j = 0

        for i in range(self.pop_size):
            if self.Fitness[i] > self.Fitness[j]:
                j = i
    
        for i in range(self.dim):
            self.pbest[i] = self.Xs[j]

        self.Fmax = self.Fitness[j]
        self.gbest = self.Xs[j]

    def salp_position(self):
        
        for i in range(self.dim):
            self.LB[i] = lower[i]
            self.UB[i] = upper[i]

        for i in range(self.pop_size):
            for j in range(self.dim):
                self.Xs[i][j] = self.LB[j] + (self.UB[j] - self.LB[j]) * np.random.uniform(0, 1)
            self.Fitness[i] = self.fitness(self.Xs[i])
        
        self.best_salp()

    def simplebounds(self, val, lower, upper):
        if val < lower:
            val = lower
        if val > upper:
            val = upper
        return val

    def fitness(self, array):
        fitness = obj(array)
        return fitness

    def run(self):
        self.salp_position()

        Convergence_curve = np.zeros(iterations)
        timerStart = time.time()

        t = 0 
        c1 = 2*np.exp(-(t/self.iterations)**2)

        for i in range(self.pop_size):
            for j in range(self.dim):
                if random.random() >= 0.5 and i < pop_size/2:
                    self.X[i][j] = self.gbest[j] + \
                        c1*((self.UB[j]-self.LB[j]) *
                            random.random() + self.LB[j])
                elif random.random() < 0.5 and i < pop_size/2:
                    self.X[i][j] = self.gbest[j] - \
                        c1*((self.UB[j]-self.LB[j]) *
                            random.random() + self.LB[j])
                else:
                    self.X[i][j] = (self.Xs[i][j] + self.Xs[i-1][j])/2

                self.X[i][j] = self.simplebounds(
                    self.X[i][j], self.LB[j], self.UB[j])
                
        t += 1

        while (t < self.iterations):

            c1 = 2*np.exp(-(t/self.iterations)**2)

            for i in range(self.pop_size):
                for j in range(self.dim):
                    M = np.sum(self.pbest, axis=0)/self.pop_size
                    #print("M:", M)
                    if random.random() >= 0.5 and i < pop_size/2:

                        self.X[i][j] = self.gbest[j] + c1 * \
                            (M[j] - self.Xs[i][j])*np.log(1/random.random())
                    elif random.random() < 0.5 and i < pop_size/2:
                        self.X[i][j] = self.gbest[j] - c1 * \
                            (M[j] - self.Xs[i][j])*np.log(1/random.random())
                    else:
                        self.Xe[i][j] = random.random(
                        )*(self.UB[j]+self.LB[j]) - self.Xs[i][j]
                        self.Xe[i][j] = self.simplebounds(
                            self.X[i][j], self.LB[j], self.UB[j])

                    self.X[i][j] = self.simplebounds(
                        self.X[i][j], self.LB[j], self.UB[j])

                # REVERSE ELITE LEARNING
                if(i > pop_size/2):
                    if self.fitness(self.Xe[i]) > self.fitness(self.Xs[i]):
                        self.X[i] = self.Xe[i]

                # WAVELET MUTATION PROCESS
                power = -np.log(self.g1)*(1-t /
                                          self.iterations)**self.epsilon + np.log(self.g1)
                sigma = np.exp(power)
                if sigma > 0:
                    for j in range(self.dim):
                        self.Xm[i][j] = self.Xs[i][j] - \
                            sigma*(self.UB[j] - self.Xs[i][j])
                        self.Xm[i][j] = self.simplebounds(
                            self.Xm[i][j], self.LB[j], self.UB[j])
                else:
                    for j in range(self.dim):
                        self.Xm[i][j] = self.Xs[i][j] - \
                            sigma*(self.Xs[i][j] - self.LB[j])
                        self.Xm[i][j] = self.simplebounds(
                            self.Xm[i][j], self.LB[j], self.UB[j])

                if self.fitness(self.Xm[i]) > self.fitness(self.X[i]):
                    self.X[i] = self.Xm[i]

            for i in range(self.pop_size):
                Fnew = self.fitness(self.X[i])

                if (Fnew >= self.Fitness[i]):
                    for j in range(self.dim):
                        self.Xs[i][j] = self.X[i][j]
                    self.pbest[i] = self.Xs[i]
                    self.Fitness[i] = Fnew

                if Fnew >= self.Fmax:
                    self.gbest = self.X[i]
                    self.Fmax = Fnew

            Convergence_curve[t] = self.Fmax
            t += 1

            print(self.Fmax)
        
        timerEnd = time.time()
        executionTime = timerEnd - timerStart
        bestIndividual = self.gbest

        # Plot the convergence curve
        plt.plot(range(1, iterations+1), Convergence_curve, label='Convergence Curve')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function - Cp')
        plt.title('Convergence Curve of QSSA')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.show()

        for i in range(300):
            print(Convergence_curve[i])

        print("Best solution:", bestIndividual)
        print("Best fitness:", self.Fmax)
        print("Execution time:", executionTime)


if __name__ == '__main__':
    result = QSSA(pop_size, dim, iterations, lower, upper)
    result.run()
    
