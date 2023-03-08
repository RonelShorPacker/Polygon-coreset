import numpy as np
import matplotlib.pyplot as plt
from set_of_points import SetOfPoints
from tests import Test
from parameters_config import ParameterConfig
from create_data import polygon_data
import pickle

def create_data(file_name):
    points = []
    for point in open(file_name).readlines():
        values = point.split(',')
        points.append([float(values[0]), float(values[2])])
    return np.array(points)

def main():
    params = ParameterConfig()
    # create data
    data = polygon_data(params.k, is_debug=True)
    # data = create_data("../../AutonomousDroneCPP/datasets/buildings/HUJI/Koma2/pointData8.csv")
    # with open('data.p', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('data.p', 'rb') as handle:
    #     data = pickle.load(handle)
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()
    P = SetOfPoints(data, parameters_config=params)
    sizes = np.linspace(20, 500, 15, dtype=int, endpoint=False)
    test = Test(P, 32, sizes)
    test.testUniformSampling(plot=True)
    test.testWeights()
    # test.testCoreset()


if __name__ == "__main__":
    main()