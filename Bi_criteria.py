import numpy as np
from parameters_config import ParameterConfig
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import null_space

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def clusterIdxsBasedOnKSubspaces(P, B):
    """
    This functions partitions the points into clusters a list of flats.
    :param B: A list of flats
    :return: A numpy array such each entry contains the index of the flat to which the point which is related to the
             entry is assigned to.
    """
    n = P.shape[0]
    idxs = np.arange(n)  # a numpy array of indices
    centers = np.array(B)  # a numpy array of the flats

    dists = np.apply_along_axis(lambda x: computeDistanceToSubspace(P[idxs, :], x[0], x[1]), 1, centers)  # compute the
                                                                                                # distance between
                                                                                                # each point and
                                                                                                # each flat
    idxs = np.argmin(dists, axis=0)

    return idxs  # return the index of the closest flat to each point in self.P.P

####################################################### Bicriteria #####################################################
parametrs_config = ParameterConfig()
LAMBDA = 1
Z = 2
M_ESTIMATOR_FUNCS = {
    'lp': (lambda x: np.abs(x) ** Z / Z),
    'huber': (lambda x: np.where(np.abs(x) <= LAMBDA, x ** 2 / 2, LAMBDA * (np.abs(x) - LAMBDA / 2))),
    'cauchy': (lambda x: LAMBDA ** 2 / 2 * np.log(1 + x ** 2 / LAMBDA ** 2)),
    'geman_McClure': (lambda x: x ** 2 / (2 * (1 + x ** 2))),
    'welsch': (lambda x: LAMBDA ** 2 / 2 * (1 - np.exp(-x ** 2 / LAMBDA ** 2))),
    'tukey': (lambda x: np.where(np.abs(x) <= LAMBDA, LAMBDA ** 2 / 6 * (1 - (1 - x ** 2 / LAMBDA ** 2) ** 3),
                                 LAMBDA**2 / 6)),
    'L1-2': (lambda x: 2 * (np.sqrt(1 + x ** 2 / 2) - 1)),
    'fair': (lambda x: LAMBDA ** 2 * (np.abs(x) / LAMBDA - np.log(1 + np.abs(x) / LAMBDA))),
    'logit': (lambda x: 1.0 / (1 + np.exp(-x))),
    'relu': (lambda x: np.max(x, 0)),
    'leaky-relu': (lambda x: x if x > 0 else 0.01 * x)
}

METHOD = 'L1-2'
OBJECTIVE_LOSS = M_ESTIMATOR_FUNCS[METHOD]  # the objective function which we want to generate a coreset for


####################################################### Bicriteria #####################################################

def computeDistanceToSubspace(point, X, v=None):
    """
    This function is responsible for computing the distance between a point and a J dimensional affine subspace.
    :param point: A numpy array representing a .
    :param X: A numpy matrix representing a basis for a J dimensional subspace.
    :param v: A numpy array representing the translation of the subspace from the origin.
    :return: The distance between the point and the subspace which is spanned by X and translated from the origin by v.
    """
    global OBJECTIVE_LOSS
    if point.ndim > 1:
        return OBJECTIVE_LOSS(np.linalg.norm(np.dot(point - v[np.newaxis, :], null_space(X)), ord=2, axis=1))
    return OBJECTIVE_LOSS(np.linalg.norm(np.dot(point-v if v is not None else point, null_space(X))))

def attainClosestPointsToSubspaces(P, W, flats, indices):
    """
    This function returns the closest n/2 points among all of the n points to a list of flats.
    :param flats: A list of flats where each flat is represented by an orthogonal matrix and a translation vector.
    :param indices: A list of indices of points in self.P.P
    :return: The function returns the closest n/2 points to flats.
    """
    dists = np.empty((P[indices, :].shape[0], ))
    N = indices.shape[0]
    if not parametrs_config.fast_bi_criteria:
        for i in range(N):
            dists[i] = np.min([
                computeDistanceToSubspace(P[np.array([indices[i]]), :], flats[j][0], flats[j][1])
                for j in range(len(flats))])

    else:
        dists = computeDistanceToSubspace(P[indices, :], flats[0], flats[1])
        idxs = np.argpartition(dists, N // 2)[:N//2]
        return idxs.tolist()

    return np.array(indices)[np.argsort(dists).astype(np.int)[:int(N / 2)]].tolist()



def sortDistancesToSubspace(P, X, v, points_indices):
    """
    The function at hand sorts the distances in an ascending order between the points and the flat denoted by (X,v).
    :param X: An orthogonal matrix which it's span is a subspace.
    :param v: An numpy array denoting a translation vector.
    :param points_indices: a numpy array of indices for computing the distance to a subset of the points.
    :return: sorted distances between the subset points addressed by points_indices and the flat (X,v).
    """
    dists = computeDistanceToSubspace(P[points_indices, :], X, v)  # compute the distance between the subset
                                                                         # of points towards
                                                                         # the flat which is represented by (X,v)
    return np.array(points_indices)[np.argsort(dists).astype(np.int)].tolist()  # return sorted distances


def computeSubOptimalFlat(P, weights):
    """
    This function computes the sub optimal flat with respect to l2^2 loss function, which relied on computing the
    SVD factorization of the set of the given points, namely P.
    :param P: A numpy matrix which denotes the set of points.
    :param weights: A numpy array of weightes with respect to each row (point) in P.
    :return: A flat which best fits P with respect to the l2^2 loss function.
    """
    v = np.average(P, axis=0, weights=weights)  # compute the weighted mean of the points
    svd = TruncatedSVD(algorithm='randomized', n_iter=1, n_components=1).fit(P-v)
    V = svd.components_
    return V, v  # return a flat denoted by an orthogonal matrix and a translation vector



def addFlats(P, W, S, B):
    """
    This function is responsible for computing a set of all possible flats which passes through j+1 points.
    :param S: list of j+1 subsets of points.
    :return: None (Add all the aforementioned flats into B).
    """
    indices = [np.arange(S[i].shape[0]) for i in range(len(S))]

    points = np.meshgrid(*indices)                    # compute a mesh grid using the duplicated coefs
    points = np.array([p.flatten() for p in points])  # flatten each point in the meshgrid for computing the
                                                      # all possible ordered sets of j+1 points
    idx = len(B)
    for i in range(points.shape[1]):
        A = [S[j][points[j, i]][0] for j in range(points.shape[0])]
        P_sub, W_sub = P[A, :], W[A]
        B.append(computeSubOptimalFlat(P_sub, W_sub))

    return np.arange(idx, len(B)), B


def computeBicriteria(P, W):
    """
    The function at hand is an implemetation of Algorithm Approx-k-j-Flats(P, k, j) at the paper
    "Bi-criteria Linear-time Approximations for Generalized k-Mean/Median/Center". The algorithm returns an
    (2^j, O(log(n) * (jk)^O(j))-approximation algorithm for the (k,j)-projective clustering problem using the l2^2
    loss function.
    :return: A (2^j, O(log(n) * (jk)^O(j)) approximation solution towards the optimal solution.
    """
    n = P.shape[0]
    Q = np.arange(0, n, 1)
    t = 0
    B = []
    tol_sample_size = parametrs_config.k * (1 + 1)
    sample_size = (lambda t: int(np.ceil(parametrs_config.k * (1 + 1) * (2 + np.log(1 + 1) + np.log(parametrs_config.k) + min(t, np.log(np.log(n)))))))

    while np.size(Q) >= tol_sample_size:  # run we have small set of points
        S = []
        for i in range(0, 1+1):  # Sample j + 1 subsets of the points in an i.i.d. fashion
            random_sample = np.random.choice(Q, size=sample_size(t))
            S.append(random_sample[:, np.newaxis])

        if not parametrs_config.fast_bi_criteria:
            a, F = addFlats(P, W, S, B)
        else:
            S = np.unique(np.vstack(S).flatten())
            F = computeSubOptimalFlat(P[S, :], W[S])
            B.append(F)

        sorted_indices = attainClosestPointsToSubspaces(P, W, F, Q)
        Q = np.delete(Q, sorted_indices)
        t += 1

    if not parametrs_config.fast_bi_criteria:
        _, B = addFlats(P, W, [Q for i in range(1 + 1)], B)
    else:
        F = computeSubOptimalFlat(P[Q.flatten(), :], W[Q.flatten()])
        B.append(F)

    return B


def projectPointsToSubspace(P, X, v):
    return np.dot(P-v if v is not None else P, null_space(X))


def applyBiCriteria(P, B, I):
    clusters = []
    for i in range(len(B)):
        F = np.array(B)
        idx = np.where(I == i)[0]
        if np.size(idx):
            P_c = P.get_points_from_indices(idx)
            P_proj = P_c.project_on_subspace(F[i])
            clusters.append([P_proj, F])
        else:
            continue

    return np.array(clusters, dtype=object)



"""def Bi_Criteria(P, k, threshold_Bi_Criteria=6):
    
    :param P: (SetofPoints) data points
    :param percentage_of_points_to_reduce_every_iteration: every iteration we will remove this percentage of points
    :param sample_size: number of points we will uniformlly choose every iteration
    :return: array of lines
    
    P_temp = P.get_copy()
    L = []

    parameters_config = ParameterConfig()
    #threshold_stop = parameters_config.minimum_points
    num_iterations = 0
    parameters_config.sample_size_bi_criteria = 100#int(32 * k * (2 + np.log(k)) + np.min([num_iterations, np.log(np.log(P_temp.get_size()))]))
    parameters_config.loop_threshold = 64 * k
    parameters_config.reduce_each_iteration = 0.5  # int(parameters_config.gamma*P.get_size())
    while P_temp.get_size() > parameters_config.loop_threshold:
        # choosing the sample
        sample = P_temp.get_sample_of_points(parameters_config.sample_size_bi_criteria)  # parameters_config.sample_size)
        # sample = np.random.choice(P_temp, sample_size)
        # making a line between every two points in the sample
        # sample_lines = get_line_between_every_two_points(sample, sample_size)
        sample_lines = sample.get_line_between_every_two_points()
        # adding the lines to the output
        L.extend(sample_lines)
        # finding the cost of every point from the lines we found
        # costs = compute_cost_for_P(P_temp, sample_lines)
        costs = P_temp.compute_distances_from_lines(sample_lines)
        # sorting the costs from smallest to biggest
        sorted_indices_by_cost = np.argsort(costs)
        # removing the percentage of points that have the lowest cost
        P_temp.remove_points(sorted_indices_by_cost, parameters_config.reduce_each_iteration)

        print(P_temp.get_size())

    return L"""