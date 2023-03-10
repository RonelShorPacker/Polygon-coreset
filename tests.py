import numpy as np
import matplotlib.pyplot as plt
from coreset import computeSensitivities, sampleCoreset
from algo import exhaustive_search, computeCost, computeCostToPolygon
from tqdm import tqdm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import pickle

class Test:
    def __init__(self, P, iterations, sizes):
        self.P = P
        self.iterations = iterations
        self.sizes = sizes

    def testUniformSampling(self, plot=False):
        epsilon_array_uniform = np.zeros((self.sizes.shape[0], self.iterations))
        epsilon_array_coreset = np.zeros((self.sizes.shape[0], self.iterations))
        print("Finding Optimal Polygon")
        with open('opt.p', 'rb') as handle:
            opt = pickle.load(handle)
        opt_cost = computeCostToPolygon(self.P, opt)
        # opt, opt_cost = exhaustive_search(self.P, iters=10000, plot=True)
        # with open('opt.p', 'wb') as handle:
        #     pickle.dump(opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Finished finding Optimal Polygon")
        for i, size in tqdm(enumerate(self.sizes), desc='Testing'):
            print(f'Testing size {size}')
            self.P.parameters_config.coreset_size = size
            P_S = computeSensitivities(self.P)
            for j in tqdm(range(self.iterations)):
                uniform_sample = P_S.get_sample_of_points(size)
                uniform_sample.weights = np.ones_like(uniform_sample.weights) * P_S.get_size() / size
                C = sampleCoreset(P_S, P_S.parameters_config.coreset_size)
                polygon_uniform, _ = exhaustive_search(uniform_sample)
                polygon_coreset, _ = exhaustive_search(C)
                # cost_uniform = computeCost(uniform_sample, polygon_uniform)
                # cost_coreset = computeCost(C, polygon_coreset)
                # cost_opt_uniform = computeCost(self.P, polygon_uniform)
                # cost_opt_coreset = computeCost(self.P, polygon_coreset)
                cost_uniform = computeCostToPolygon(uniform_sample, polygon_uniform)
                cost_coreset = computeCostToPolygon(C, polygon_coreset)
                cost_opt_uniform = computeCostToPolygon(self.P, polygon_uniform)
                cost_opt_coreset = computeCostToPolygon(self.P, polygon_coreset)
                epsilon_array_uniform[i][j] = self.computeEpsilon(cost_opt_uniform, cost_uniform)
                epsilon_array_coreset[i][j] = self.computeEpsilon(cost_opt_coreset, cost_coreset)
            print(f'uniform = {epsilon_array_uniform[i].mean()}')
            print(f'coreset = {epsilon_array_coreset[i].mean()}')

        max_epsilon_array_uniform = np.max(epsilon_array_uniform, axis=1)
        min_epsilon_array_uniform = np.min(epsilon_array_uniform, axis=1)
        mean_epsilon_array_uniform = np.mean(epsilon_array_uniform, axis=1)
        max_epsilon_array_coreset = np.max(epsilon_array_coreset, axis=1)
        min_epsilon_array_coreset = np.min(epsilon_array_coreset, axis=1)
        mean_epsilon_array_coreset = np.mean(epsilon_array_coreset, axis=1)


        plt.title("Coreset vs Uniform")
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
                C = sampleCoreset(P_S, P_S.parameters_config.coreset_size)
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
        return np.abs((opt_cost - coreset_cost))/(opt_cost)


    def testWeights(self):
        epsilon_array_weights = np.zeros((self.sizes.shape[0], self.iterations))


        for i, size in tqdm(enumerate(self.sizes), desc=f'Testing'):
            self.P.parameters_config.coreset_size = size
            P_S = computeSensitivities(self.P)
            for j in range(self.iterations):
                C = sampleCoreset(P_S, P_S.parameters_config.coreset_size)
                epsilon_array_weights[i][j] = self.computeEpsilon(self.P.get_size(), C.weights.sum())
        mean_epsilon_array_coreset = np.mean(epsilon_array_weights, axis=1)

        plt.title("Weights")
        plt.xlabel("Coreset size")
        plt.ylabel("Epsilon")
        plt.plot(self.sizes, mean_epsilon_array_coreset, label="sum of weights")
        plt.legend()
        plt.grid()
        plt.show()

    def testCoreset(self):
        print("Sampling Polygons")
        polygons = self.samplePolygons(10)
        print("Finished sampling Polygons")

        epsilon_array_uniform = np.zeros((len(polygons), self.sizes.shape[0], self.iterations))
        epsilon_array_coreset = np.zeros((len(polygons), self.sizes.shape[0], self.iterations))

        for l, polygon in enumerate(polygons):
            for i, size in tqdm(enumerate(self.sizes), desc='Testing'):
                print(f'Testing size {size}')
                self.P.parameters_config.coreset_size = size
                P_S = computeSensitivities(self.P)
                for j in tqdm(range(self.iterations)):
                    uniform_sample = P_S.get_sample_of_points(size)
                    uniform_sample.weights = np.ones_like(uniform_sample.weights) * P_S.get_size() / size
                    C = sampleCoreset(P_S, P_S.parameters_config.coreset_size)
                    cost_uniform = computeCostToPolygon(uniform_sample, polygon)
                    cost_coreset = computeCostToPolygon(C, polygon)
                    cost_opt_uniform = computeCostToPolygon(self.P, polygon)
                    cost_opt_coreset = computeCostToPolygon(self.P, polygon)
                    epsilon_array_uniform[l][i][j] = self.computeEpsilon(cost_opt_uniform, cost_uniform)
                    epsilon_array_coreset[l][i][j] = self.computeEpsilon(cost_opt_coreset, cost_coreset)

            max_epsilon_array_uniform = np.max(epsilon_array_uniform[l], axis=1)
            min_epsilon_array_uniform = np.min(epsilon_array_uniform[l], axis=1)
            mean_epsilon_array_uniform = np.mean(epsilon_array_uniform[l], axis=1)
            max_epsilon_array_coreset = np.max(epsilon_array_coreset[l], axis=1)
            min_epsilon_array_coreset = np.min(epsilon_array_coreset[l], axis=1)
            mean_epsilon_array_coreset = np.mean(epsilon_array_coreset[l], axis=1)

            plt.title("Coreset vs Uniform, Random Polygon")
            plt.xlabel("Coreset size")
            plt.ylabel("Epsilon")
            plt.plot(self.sizes, mean_epsilon_array_uniform, label="Uniform")
            plt.plot(self.sizes, mean_epsilon_array_coreset, label="Coreset")
            plt.legend()
            plt.grid()
            plt.show()


    def samplePolygons(self, num):
        polygons = []
        count = 0
        while count < num:
            sample = sampleCoreset(self.P, self.P.parameters_config.k)
            try:
                convex_hull = ConvexHull(sample.points)
            except:
                continue

            if len(convex_hull.vertices) != self.P.parameters_config.k:
                continue

            polygons.append(convex_hull)
            count += 1
        return polygons