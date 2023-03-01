import numpy as np
import matplotlib.pyplot as plt
from coreset import computeSensitivities, sampleCoreset
from algo import exhaustive_search, computeCost
from tqdm import tqdm
from scipy.spatial import convex_hull_plot_2d

class Test:
    def __init__(self, P, iterations, sizes):
        self.P = P
        self.iterations = iterations
        self.sizes = sizes

    def testUniformSampling(self, plot=False):
        epsilon_array_uniform = np.zeros((self.sizes.shape[0], self.iterations))
        epsilon_array_coreset = np.zeros((self.sizes.shape[0], self.iterations))
        print("Finding Optimal Polygon")
        opt, opt_cost = exhaustive_search(self.P, plot=True)
        print("Finished finding Optimal Polygon")
        P_S = computeSensitivities(self.P)
        for i, size in tqdm(enumerate(self.sizes), desc='Testing'):
            print(f'Testing size {size}')
            P_S.parameters_config.coreset_size = size
            for j in tqdm(range(self.iterations)):
                uniform_sample = P_S.get_sample_of_points(size)
                C = sampleCoreset(P_S)
                polygon_uniform, _ = exhaustive_search(uniform_sample)
                polygon_coreset, _ = exhaustive_search(C)
                cost_uniform = computeCost(P_S, polygon_uniform)
                cost_coreset = computeCost(P_S, polygon_coreset)
                epsilon_array_uniform[i][j] = self.computeEpsilon(opt_cost, cost_uniform)
                epsilon_array_coreset[i][j] = self.computeEpsilon(opt_cost, cost_coreset)

        max_epsilon_array_uniform = np.max(epsilon_array_uniform, axis=1)
        min_epsilon_array_uniform = np.min(epsilon_array_uniform, axis=1)
        mean_epsilon_array_uniform = np.mean(epsilon_array_uniform, axis=1)
        max_epsilon_array_coreset = np.max(epsilon_array_coreset, axis=1)
        min_epsilon_array_coreset = np.min(epsilon_array_coreset, axis=1)
        mean_epsilon_array_coreset = np.mean(epsilon_array_coreset, axis=1)


        plt.title("Coreset vs Optimum")
        plt.xlabel("Coreset size")
        plt.ylabel("Epsilon")
        plt.plot(self.sizes, mean_epsilon_array_uniform, label="Uniform")
        plt.plot(self.sizes, mean_epsilon_array_coreset, label="Coreset")
        plt.legend()
        plt.grid()
        plt.show()

    def testvsOpt(self):
        epsilon_array = np.zeros((self.sizes.shape[0], self.iterations))
        print("Finding Optimal Polygon")
        opt, opt_cost = exhaustive_search(self.P, 10)
        print("Found Optimal Polygon")
        P_S = computeSensitivities(self.P)

        for i, size in tqdm(enumerate(self.sizes), desc=f'Testing'):
            P_S.parameters_config.coreset_size = size
            for j in range(self.iterations):
                C = sampleCoreset(P_S)
                C_opt, _ = exhaustive_search(C)
                polygon_coreset, _ = exhaustive_search(C)
                cost_coreset = computeCost(P_S, polygon_coreset)
                epsilon_array[i][j] = self.computeEpsilon(opt_cost, cost_coreset)

        max_epsilon_array = np.max(epsilon_array, axis=1)
        min_epsilon_array = np.min(epsilon_array, axis=1)
        mean_epsilon_array = np.mean(epsilon_array, axis=1)

        plt.title("Coreset vs Optimum")
        plt.xlabel("coreset size")
        plt.ylabel("epsilon")
        plt.plot(self.sizes, mean_epsilon_array)
        plt.grid()
        plt.show()

    def computeEpsilon(self, opt_cost, coreset_cost):
        return np.abs((opt_cost - coreset_cost)/(opt_cost))


