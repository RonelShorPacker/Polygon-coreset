import numpy as np
import matplotlib.pyplot as plt
from main import coreset


class Test:
    def __init__(self, P, iterations):
        self.P = P
        self.iterations = iterations
    def testUniformSampling(self):
        C = coreset(self.P)
        for i in range(self.iterations):
            uniform_sample_indices = np.random.uniform(self.P.get_size(), size=C.get_size())
            uniform_sample = self.P[uniform_sample_indices, :]