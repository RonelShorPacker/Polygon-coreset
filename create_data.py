import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def generate_polygon(center, avg_radius, irregularity, spikiness, num_vertices):
    """
        Start with the center of the polygon at center, then creates the
        polygon by sampling points on a circle around the center.
        Random noise is added by varying the angular spacing between
        sequential points, and by varying the radial distance of each
        point from the centre.

        Args:
            center (Tuple[float, float]):
                a pair representing the center of the circumference used
                to generate the polygon.
            avg_radius (float):
                the average radius (distance of each generated vertex to
                the center of the circumference) used to generate points
                with a normal distribution.
            irregularity (float):
                variance of the spacing of the angles between consecutive
                vertices.
            spikiness (float):
                variance of the distance of each vertex to the center of
                the circumference.
            num_vertices (int):
                the number of vertices of the polygon.
        Returns:
            List[Tuple[float, float]]: list of vertices, in CCW order.
        """
    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        if i != 0 and i != 1:
            radius = np.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
            point = (center[0] + radius * np.cos(angle),
                     center[1] + radius * np.sin(angle))
            points.append(point)
            angle += angle_steps[i]
        else:
            radius = 2200 * np.clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
            point = (center[0] + radius * np.cos(angle),
                     center[1] + radius * np.sin(angle))
            points.append(point)
            angle += angle_steps[i]
    return points


def random_angle_steps(steps, irregularity):
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def polygon_data(k, with_partition=False, is_debug=False):
    polygon = np.array(generate_polygon([0, 0], 12, 0.4, 0.4, k))
    # opt = ConvexHull(polygon)
    if with_partition:
        data_groups = {'segment_groups': [], 'vertex_groups': []}
    else:
        data_groups = []

    # segment groups
    for i in range(k):
        if i == k-1:
            num_points = np.random.randint(1500, 2000)
            # noise = np.random.normal(0, 1, (num_points, 2))
            noise = np.random.normal(0, 0, (num_points, 2))
            lam = np.random.uniform(0.8, 1, num_points)
            points = []
            for j in range(num_points):
                points.append(lam[j] * polygon[i] + (1 - lam[j]) * polygon[(i + 1) % k] + noise[j])

            if with_partition:
                data_groups['segment_groups'].append(np.array(points))
            else:
                data_groups.append(np.array(points))
        else:
            if i != 0:
                num_points = np.random.randint(1500, 2000)
                # noise = np.random.normal(0, 1, (num_points, 2))
                noise = np.random.normal(0, 0, (num_points, 2))
                lam = np.random.uniform(0, 0.2, num_points)
                points = [polygon[0], polygon[1], polygon[2]]
                for j in range(num_points):
                    points.append(lam[j] * polygon[i] + (1 - lam[j]) * polygon[(i + 1)%k] + noise[j])

                if with_partition:
                    data_groups['segment_groups'].append(np.array(points))
                else:
                    data_groups.append(np.array(points))
            else:
                num_points = np.random.randint(30, 50)
                # noise = np.random.normal(0, 1, (num_points, 2))
                noise = np.random.normal(0, 0, (num_points, 2))
                lam = np.random.uniform(0.45, 0.55, num_points)
                points = []
                for j in range(num_points):
                    points.append(lam[j] * polygon[i] + (1 - lam[j]) * polygon[(i + 1) % k] + noise[j])

                if with_partition:
                    data_groups['segment_groups'].append(np.array(points))
                else:
                    data_groups.append(np.array(points))

    # vertex groups
    # for i in range(k):
    #     num_points = np.random.randint(20, 70)
    #     # noise = np.random.normal(0, 0.5, (num_points, 2))
    #     noise = np.random.normal(0, 0, (num_points, 2))
    #     lam1 = np.random.uniform(1, 1.25, num_points)
    #     lam2 = np.random.uniform(1, 1.25, num_points)
    #     lam3 = np.random.uniform(0.25, 0.75, num_points)
    #     points = []
    #     for j in range(num_points):
    #         temp1 = lam1[j] * polygon[i] + (1 - lam1[j]) * polygon[(i - 1)%k]
    #         temp2 = lam2[j] * polygon[i] + (1 - lam2[j]) * polygon[(i + 1)%k]
    #         points.append(lam3[j] * temp1 + (1 - lam3[j]) * temp2 + noise[j])
    #
    #     if with_partition:
    #         data_groups['vertex_groups'].append(np.array(points))
    #     else:
    #         data_groups.append(np.array(points))

    if not with_partition:
        data_groups = np.concatenate(data_groups, axis=0)

    if is_debug:
        if with_partition:
            for i in range(len(polygon)):
                plt.plot([polygon[i][0], polygon[(i+1)%len(polygon)][0]], [polygon[i][1], polygon[(i+1)%len(polygon)][1]], c='red')
            for groups in data_groups.values():
                for group in groups:
                    plt.scatter(group[:, 0], group[:, 1])
            plt.show()
        else:
            for i in range(len(polygon)):
                plt.plot([polygon[i][0], polygon[(i+1)%len(polygon)][0]], [polygon[i][1], polygon[(i+1)%len(polygon)][1]], c='red')
            plt.scatter(data_groups[:, 0], data_groups[:, 1])
            plt.show()

    return data_groups

