import numpy as np
import matplotlib.pyplot as plt
from main import coreset


class Test:
    def __init__(self, P, iterations, sizes):
        self.P = P
        self.iterations = iterations
        self.sizes = sizes


    def computeEpsilon(self):
        pass

    def testUniformSampling(self):

        C = coreset(self.P, )
        for i in range(self.iterations):
            uniform_sample_indices = np.random.uniform(self.P.get_size(), size=C.get_size())
            uniform_sample = self.P[uniform_sample_indices, :]





    def testvsOpt(self):
        epsilon_array = np.zeros((self.sizes.shape[0], self.iterations))
        opt, opt_cost = exhaustive_search(self.P)
        P_S = coreset(self.P)

        for i, size in enumerate(self.sizes):
            self.P.parameters_config.coreset_size = size
            for j in tqdm(range(self.iterations)):
                C = sample_coreset(self.P)
                C_opt = exhaustive_search(C)
                C_cost =
                epsilon_array[i][j] = self.computeEpsilon(opt_cost, C_cost)

        max_epsilon_array = np.max(epsilon_array, axis=0)
        min_epsilon_array = np.min(epsilon_array, axis=0)