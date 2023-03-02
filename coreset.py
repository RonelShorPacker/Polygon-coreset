import numpy as np
from set_of_points import SetOfPoints
from Bi_criteria import computeBicriteria, clusterIdxsBasedOnKSubspaces, applyBiCriteria


def sampleCoreset(P, sample_size):
    probs = P.get_probabbilites().reshape(1, -1)[0]
    indices_sample = np.random.choice(np.arange(P.get_size()), sample_size, True, probs.astype(float))
    A = P.points[indices_sample]
    v = P.weights[indices_sample].reshape(1, -1)[0]
    s = P.sensitivities[indices_sample].reshape(1, -1)[0]
    return SetOfPoints(A, v, s)

def computeSensitivities(P):
    """
    Args:
        P: SetofPoints
    Returns: SetofPoints, coreset or np.array, sensitivities

    """

    # computing Bi-criteria
    B = computeBicriteria(P.points, np.squeeze(P.weights))
    # Project data to Bi criteria
    indices = clusterIdxsBasedOnKSubspaces(P.points, B)
    P_proj = applyBiCriteria(P, B, indices)

    # for i in range(len(P_proj)):
    #     plt.scatter(P_proj[i][0].points[:, 0], P_proj[i][0].points[:, 1])
    # plt.show()

    P_S = SetOfPoints(parameters_config=P.parameters_config)

    # compute sensitivity per cluster
    for cluster in P_proj[:, 0]:
        sen = cluster.computeSensativities()
        orig_P = P.get_points_from_indices(cluster.indexes).points
        tmp = SetOfPoints(P=orig_P, sen=sen, indexes=cluster.indexes)
        P_S.add_set_of_points(tmp)
    P_S.set_weights(P_S.get_sum_of_sensitivities(), P_S.parameters_config.coreset_size)

    return P_S
