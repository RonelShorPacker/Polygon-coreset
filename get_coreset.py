from set_of_points import SetOfPoints
from parameters_config import ParameterConfig
from coreset import computeSensitivities, sampleCoreset
from numpy import zeros, savetxt

import sys


def create_data(file_name):
    points = []
    for point in open(file_name).readlines():
        values = point.split(',')
        points.append([float(values[0]), float(values[2])])
    return points


def from_list_to_ndarray(points):
    data = zeros((len(points), 2))
    for i, point in enumerate(points):
        data[i, 0] = point[0]
        data[i, 1] = point[1]
    return data


def getCoresetIndexes(data_path, size):
    """
    :param data_path : path of data: numpy nxd array, represents points in R^d
    :param size: size of wanted coreset
    :return: SetofPoints, the coreset
    """
    data = from_list_to_ndarray(create_data(data_path))
    # parameters for the coreset
    params = ParameterConfig()
    # defining class for data
    P = SetOfPoints(data, parameters_config=params)
    # compute sensitivity of each point
    P_S = computeSensitivities(P)
    # defined size of coreset as defined by user
    P_S.parameters_config.coreset_size = size
    # define weights from the sensitivities
    P_S.set_weights(P_S.get_sum_of_sensitivities(), P_S.parameters_config.coreset_size)
    # sample coreset
    C = sampleCoreset(P_S, P_S.parameters_config.coreset_size)

    return C.indexes


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_path> <size> <output_path>")
        sys.exit(1)

    # Extract the command-line arguments
    data_path = sys.argv[1]
    size = int(sys.argv[2])
    output_path = sys.argv[3]
    indexes = getCoresetIndexes(data_path, size)
    print(indexes)
    # Write indexes to the output file
    savetxt(output_path, indexes, delimiter=',', fmt='%d')
    print("Integration Works!")

