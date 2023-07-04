from set_of_points import SetOfPoints
from parameters_config import ParameterConfig
from coreset import computeSensitivities, sampleCoreset


def getCoreset(data, size):
    """
    :param data: numpy nxd array, represents points in R^d
    :param size: size of wanted coreset
    :return: SetofPoints, the coreset
    """
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

    return C


