import numpy as np
import matplotlib.pyplot as plt
from main import coreset, sampleCoreset
from algo import exhaustive_search

class Test:
    def __init__(self, P, iterations):
        self.P = P
        self.iterations = iterations
    def testUniformSampling(self):
        sen = coreset(self.P, sen=True)
        for i in range(self.iterations):
            uniform_sample = self.P.get_sample_of_points(C.get_size())
            C = sampleCoreset()
            exhaustive_search(uniform_sample)