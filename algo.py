import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from tqdm import tqdm
from coreset import sampleCoreset

def computeCostToPolygon(P, polygon):
    # segments = polygon.vertices - np.roll(polygon.vertices, shift=1))
    # segments_norms =
    # P.points = np.stack((np.mean(P.points, axis=0), np.mean(P.points, axis=0)))
    P_rep = np.repeat(P.points, P.get_size() * [polygon.vertices.shape[0]], axis=0)
    polygon_vertices_rep = np.tile(polygon.points[polygon.vertices], reps=(P.get_size(), 1))
    polygon_segments_diff = np.roll(polygon.points[polygon.vertices], shift=-1, axis=0) - polygon.points[polygon.vertices]
    polygon_segments_diff_rep = np.tile(polygon_segments_diff, reps=(P.get_size(), 1))
    polygon_segments_norm_rep = np.square(polygon_segments_diff_rep).sum(axis=1)
    dot_prod = np.sum(np.multiply(P_rep - polygon_vertices_rep, polygon_segments_diff_rep), axis=1)
    tmp = np.divide(dot_prod, polygon_segments_norm_rep)
    t = np.maximum(np.zeros_like(tmp, dtype=float), np.minimum(np.ones_like(tmp, dtype=float), tmp))
    proj = polygon_vertices_rep + np.multiply(np.tile(t, reps=(polygon_segments_diff_rep.shape[1],1)).T, polygon_segments_diff_rep)
    proj_split = np.array(np.array_split(proj, P.get_size()))
    P_rep_split = np.array(np.array_split(P_rep, P.get_size()))

    D = np.linalg.norm(proj_split - P_rep_split, axis=-1)
    index_min = np.argmin(D, axis=-1)
    proj_min = np.zeros_like(P.points)

    for i in range(P.get_size()):
        proj_min[i] = proj_split[i, index_min[i]]

    sum_ = P.weights.squeeze().dot(np.linalg.norm(P.points - proj_min, axis=1))
    return sum_
    # plt.scatter(proj_min[:, 0], proj_min[:, 1], color='black')

    # plt.scatter(P.points[:, 0], P.points[:, 1], color="orange")
    # plt.scatter(proj[:, 0], proj[:, 1])
    # plt.scatter(P.points[:, 0], P.points[:, 1], color='black')
    # plt.scatter(polygon.points[polygon.vertices, 0], polygon.points[polygon.vertices, 1])
    # plt.plot(polygon.points[polygon.vertices, 0], polygon.points[polygon.vertices, 1], 'r--', lw=2)
    # plt.show()
    # print("a")

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
        count += 1
        sample = sampleCoreset(P, P.parameters_config.k)
        try:
            convex_hull = ConvexHull(sample.points)
        except:
            continue

        if len(convex_hull.vertices) != P.parameters_config.k:
            continue

        # sum_ = computeCost(P, convex_hull)
        sum_ = computeCostToPolygon(P, convex_hull)
        # compute distance of each point to polygon
        if sum_ < min_sum:
            min_sum = sum_
            min_polygon = convex_hull

    if plot:
        convex_hull_plot_2d(min_polygon)
        plt.scatter(P.points[:, 0], P.points[:, 1])
        plt.show()

    return min_polygon, min_sum