import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from tqdm import tqdm
from coreset import sampleCoreset


def min_distance(pt1, pt2, p):
    l = np.sum((pt2-pt1)**2)
    t = np.max([0., np.min([1., np.dot(p-pt1, pt2-pt1) /l])])
    proj = pt1 + t*(pt2-pt1)
    return proj, np.sum((proj-p)**2)


def dist_to_polygon(P, p, polygon):
    min_dist_ = np.inf
    min_pt = None
    for i in range(len(polygon.vertices)):
        pt_proj, d = min_distance(P.points[polygon.vertices[i]], P.points[polygon.vertices[(i + 1) % len(polygon.vertices)]], p)
        if min_dist_ > d:
            min_dist_ = d
            min_pt = pt_proj
    return np.linalg.norm(p - min_pt, ord=2)


def computeCost(P, polygon):
    sum_ = 0
    for p, w in zip(P.points, P.weights):
        sum_ += w * dist_to_polygon(P, p, polygon)
    return sum_


def exhaustive_search(P, iters=1000, plot=False):
    """
    Args:
        P: SetofPoints
        iters: int, number of iterations of exhaustive search

    Returns: polygon
    """
    min_sum = np.inf
    min_polygon = None
    count = 0
    while count < iters:
        sample = sampleCoreset(P, P.parameters_config.k)#P.get_sample_of_points(P.parameters_config.k)
        try:
            convex_hull = ConvexHull(sample.points)
        except:
            continue

        if len(convex_hull.points) != P.parameters_config.k:
            continue

        sum_ = computeCost(P, convex_hull)
        # compute distance of each point to polygon
        if sum_ < min_sum:
            min_sum = sum_
            min_polygon = convex_hull
        count += 1

    if plot:
        convex_hull_plot_2d(min_polygon)
        plt.scatter(P.points[:, 0], P.points[:, 1])
        plt.show()

    return min_polygon, min_sum