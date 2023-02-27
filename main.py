import numpy as np
import matplotlib.pyplot as plt
from Bi_criteria import computeBicriteria, clusterIdxsBasedOnKSubspaces, applyBiCriteria
from set_of_points import SetOfPoints

from parameters_config import ParameterConfig
from create_data import polygon_data


def coreset(P):
    """
    Args:
        P: SetofPoints,
        W: np.array, weight of each point
    Returns: SetofPoints, coreset

    """

    # computing Bi-criteria
    B = computeBicriteria(P.points, np.squeeze(P.weights))
    # Project data to Bi criteria
    indices = clusterIdxsBasedOnKSubspaces(P.points, B)
    P_proj = applyBiCriteria(P, B, indices)

    # for i in range(len(P_proj)):
    #     plt.scatter(P_proj[i][0].points[:, 0], P_proj[i][0].points[:, 1])
    # plt.show()
    print("a")


def main():
    params = ParameterConfig()
    # create data
    data = polygon_data(params.k, is_debug=True)
    P = SetOfPoints(data)
    # extract coreset
    A = coreset(P)




if __name__ == "__main__":
    main()