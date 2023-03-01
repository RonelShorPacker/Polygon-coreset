import numpy as np
import matplotlib.pyplot as plt
from coreset import computeSensitivities, sampleCoreset
from algo import exhaustive_search, computeCost
from tqdm import tqdm

class Test:
    def __init__(self, P, iterations, sizes):
        self.P = P
        self.iterations = iterations
        self.sizes = sizes

    def testUniformSampling(self):
        P_S = computeSensitivities(self.P)
        for i, size in enumerate(self.sizes):
            P_S.parameters_config.coreset_size = size
            for j in range(self.iterations):
                uniform_sample = P_S.get_sample_of_points(P_S)
                C = sampleCoreset(P_S)
                polygon_uniform, _ = exhaustive_search(uniform_sample)
                polygon_coreset, _ = exhaustive_search(C)
                cost_uniform = computeCost(P_S, polygon_uniform)
                cost_coreset = computeCost(P_S, polygon_coreset)


        plt.title("Coreset vs Optimum")
        plt.xlabel("Coreset size")
        plt.ylabel("Epsilon")
        plt.scatter(self.sizes, )
        plt.grid()
        plt.show()

    def testvsOpt(self):
        epsilon_array = np.zeros((self.sizes.shape[0], self.iterations))
        opt, opt_cost = exhaustive_search(self.P)
        P_S = computeSensitivities(self.P)

        for i, size in enumerate(self.sizes):
            P_S.parameters_config.coreset_size = size
            for j in tqdm(range(self.iterations)):
                C = sampleCoreset(P_S)
                C_opt, _ = exhaustive_search(C)
                polygon_coreset, _ = exhaustive_search(C)
                cost_coreset = computeCost(P_S, polygon_coreset)
                epsilon_array[i][j] = self.computeEpsilon(opt_cost, cost_coreset)

        max_epsilon_array = np.max(epsilon_array, axis=0)
        min_epsilon_array = np.min(epsilon_array, axis=0)
        mean_epsilon_array = np.mean(epsilon_array, axis=0)

        plt.title("Coreset vs Optimum")
        plt.xlabel("coreset size")
        plt.ylabel("epsilon")
        plt.scatter(self.sizes, mean_epsilon_array)
        plt.grid()
        plt.show()

    def computeEpsilon(self, opt_cost, coreset_cost):
        return np.abs((opt_cost - coreset_cost)/(opt_cost))


