#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################



from __future__ import division

import copy
import random

import numpy as np
import sys

import matplotlib.pyplot as plt

from parameters_config import ParameterConfig
class SetOfPoints:
    """
    Class that represent a set of weighted points in any d>0 dimensional space
    Attributes:
        points (ndarray) : The points in the set
        weights (ndarray) : The weights. weights[i] is the weight of the i-th point
        dim (integer): The dimension of the points
    """

    ##################################################################################

    def __init__(self, P=None, w=None, sen=None, indexes = None, parameters_config = None):
        """
        C'tor
        :param P: np.ndarray - set of points
        :param w: np.ndarray - set of weights
        :param sen: np.ndarray - set of sensitivities
        :param parameters_config: class - The parameters we are using
        """
        #if (indexes != [] and len(P) == 0) or (indexes == [] and len(P) != 0):
        #    assert indexes != [] and len(P) != 0, "not indexes == [] and len(P) == 0"
        if parameters_config:
            self.parameters_config = parameters_config
        else:
            self.parameters_config = ParameterConfig()
        if P is None:
            P = []
        if w is None:
            w = []
        if sen is None:
            sen = []
        if indexes is None:
            indexes = []

        size = len(P)
        if size == 0:  # there is no points in the set we got
            self.points = []
            self.weights = []
            self.sensitivities = []
            self.dim = 0
            self.indexes = []
            return
        if np.ndim(P) == 1:  # there is only one point in the array
            Q = []
            Q.append(P)
            self.points = np.asarray(Q)
            if w == []:
                w = np.ones((1, 1), dtype=np.float)
            if sen == []:
                sen = np.ones((1, 1), dtype=np.float)
            self.weights = w
            self.sensitivities = sen
            [_, self.dim] = np.shape(self.points)
            self.indexes = np.zeros((1, 1), dtype=np.float)
            return
        else:
            self.points = np.asarray(P)
        [_, self.dim] = np.shape(self.points)
        if w == []:
            w = np.ones((size, 1), dtype=np.float)
        if sen == []:
            sen = np.ones((size, 1), dtype=np.float)
        self.weights = w
        self.sensitivities = sen
        if indexes == []:
            self.indexes = np.asarray(range(len(self.points))).reshape(-1)
        else:
            self.indexes = indexes.reshape(-1)

    ##################################################################################

    def get_sample_of_points(self, size_of_sample):
        """
        Args:
            size_of_sample (int) : the sample's size

        Returns:
            SetOfPoints: sample consist of size_of_sample points from the uniform distribution over the set
        """

        assert size_of_sample > 0, "size_of_sample <= 0"

        size = self.get_size()
        if size_of_sample >= size:
            return self
        else:
            all_indices = np.asarray(range(size))
            sample_indices = np.random.choice(all_indices, size_of_sample).tolist()
            sample_points = np.take(self.points, sample_indices, axis=0, out=None, mode='raise')
            sample_weights = np.take(self.weights, sample_indices, axis=0, out=None, mode='raise')
            sample_indexes = np.take(self.indexes, sample_indices, axis=0, out=None, mode='raise')
            return SetOfPoints(sample_points, sample_weights,indexes=sample_indexes)


    ###################################################################################
    def get_size(self):
        """
        Returns:
            int: number of points in the set
        """
        try:
            return np.shape(self.points)[0]
        except:
            print("a")

    ##################################################################################

    def get_points_from_indices(self, indices):
        """
        Args:
            indices (list of ints) : list of indices.

        Returns:
            SetOfPoints: a set of point that contains the points in the input indices
        """
        if self.get_size() == 0:
            x=2
        assert len(indices) > 0, "indices length is zero"


        sample_points = self.points[indices]
        sample_weights = self.weights[indices]
        sample_indexes = self.indexes[indices]

        return SetOfPoints(sample_points, sample_weights, indexes=sample_indexes, parameters_config=self.parameters_config)

    ##################################################################################

    def project_on_subspace(self, F):
        """
        :param F: np.ndarray, represents a subspace
        :return: SetofPoints: a set of points that contains the projected points on F
        """
        Proj = F[0] * F[0].T
        P_proj = Proj.dot((self.points - F[1]).T).T + F[1]
        return SetOfPoints(P_proj, self.weights, indexes=self.indexes, parameters_config=self.parameters_config)

    ##################################################################################

    def get_mean(self):
        """
        Returns:
            np.ndarray: the weighted mean of the points in the set
        """

        size = self.get_size()
        assert size > 0, "set is empty"

        sum_of_weights = self.get_sum_of_weights()
        points_mul_weights = sum(np.multiply(self.weights.reshape(-1, 1), self.points)) #self.points[1]*self.weights[1]+self.points[2]*self.weights[2]+...+self.points[n]*self.weights[n]
        the_mean = points_mul_weights / sum_of_weights #divition by the sum of weights to get a weighted average
        return the_mean

    ##################################################################################

    def get_sum_of_weights(self):
        """
        Returns:
            float: the sum of wights in the set
        """

        assert self.get_size() > 0, "No points in the set"

        return np.sum(self.weights)

    ##################################################################################

    def add_set_of_points(self, P):
        """
        The method adds a set of weighted points to the set
        Args:
            P (SetOfPoints) : a set of points to add to the set

        Returns:
            ~
        """

        if P.get_size() == 0:
            return

        points = P.points
        weights = P.weights.reshape(-1, 1)
        try:
            sensitivities = P.sensitivities.reshape(-1, 1)
        except:
            print("a")
        indexes = P.indexes.reshape(-1)

        size = self.get_size()
        if size == 0 and self.dim == 0:
            self.dim = np.shape(points)[1]
            self.points = points
            self.weights = weights
            self.sensitivities = sensitivities
            self.indexes = indexes
            return

        self.points = np.append(self.points, points, axis=0)
        self.weights = np.append(self.weights, weights)
        self.sensitivities = np.append(self.sensitivities, sensitivities, axis=0)
        self.indexes = np.append(self.indexes, indexes, axis=0)


    ##################################################################################

    def set_all_weights_to_specific_value(self, value):
        self.weights = np.ones(self.get_size()) * value

    ##################################################################################

    def remove_points_at_indexes(self, start, end):
        """
        TODO: complete
        :param start:
        :param end:
        :return:
        """
        indexes = np.arange(start, end)
        self.points = np.delete(self.points, indexes, axis=0)
        self.weights = np.delete(self.weights, indexes, axis=0)
        self.sensitivities = np.delete(self.sensitivities, indexes, axis=0)
        self.indexes = np.delete(self.indexes, indexes, axis=0)

    ##################################################################################

    def remove_from_set(self, C):
        """
        The method gets set of points C and remove each point in the set that also in C
        Args:
            C (SetOfPoints) : a set of points to remove from the set

        Returns:
            ~
        """

        indexes = []
        C_indexes = C.indexes
        self_indexes = self.indexes
        for i in range(len(self_indexes)):
            index = self_indexes[i]
            if index in C.indexes:
                indexes.append(i)
        #indexes = C.indexes
        self.points = np.delete(self.points, indexes, axis=0)
        self.weights = np.delete(self.weights, indexes, axis=0)
        self.sensitivities = np.delete(self.sensitivities, indexes, axis=0)
        self.indexes = np.delete(self.indexes, indexes, axis=0)

    ##################################################################################

    def get_sum_of_sensitivities(self):
        """
        Returns:
            float: the sum of the sensitivities of the points in the set
        """

        assert self.get_size() > 0, "Set is empty"

        return sum(self.sensitivities)

    ##################################################################################

    def get_closest_points_to_point(self, point, m, type):
        """
        Args:
            point (np.ndarray) : d-dimensional point
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"

        Returns:
            SetOfPoints: the points that are closest to the given point, by rate or by
                         fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        size = self.get_size()

        #this calculates all the l2 squared distances from the given point to each point in the set
        point_repeat = np.repeat(point.reshape(1, -1), repeats=size, axis=0).reshape(-1, self.dim) #this duplicate the point n times where n is the number of points in set
        the_substract = point_repeat - self.points #substract each coordinate from its corresponding coordinate
        the_multiply = (np.multiply(the_substract, the_substract))#squared of each one of the substract results
        the_plus = np.sum(the_multiply, axis=1)#sum all the squared substractions
        all_distances = np.multiply(self.weights.reshape(-1), the_plus.reshape(-1))#multiply all the distances by the weights of wach corresponding point in the set
        if type == "by rate":
            m = int(m * self.get_size()) #number of points is m percents of n
        m_th_distance = np.partition(all_distances, m)[m] #the m-th distance
        distances_smaller_than_median_indices = np.where(all_distances <= m_th_distance) #all the m smallest distances indices in self.points
        P_subset = self.points[distances_smaller_than_median_indices]
        w_subset = self.weights[distances_smaller_than_median_indices]
        indexes_subset = self.indexes[distances_smaller_than_median_indices]
        return SetOfPoints(P_subset, w_subset, indexes=indexes_subset)

    ##################################################################################

    def get_closest_points_to_set_of_points(self, P, m, type):
        """
        Args:
            P (SetOfPoints) : a set of points
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"

        Returns:
            SetOfPoints: the points that are closest to the given set of points, by rate or by
                         fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        self_size = self.get_size()
        self_points = np.asarray(self.points)
        self_weights = np.asarray(self.weights)
        P_size = P.get_size()
        P_points = np.asarray(P.points)
        P_weights = np.asarray(P.weights)

        self_points_repeat_each_point = np.repeat(self_points, repeats=P_size, axis=0) #this duplicate the self_point P_size times
        P_points_repeat_all = np.repeat(P_points.reshape(1, -1), repeats=self_size, axis=0).reshape(-1, self.dim) #this duplicate the P_points self_size times

        self_weights_repeat_each_point = np.repeat(self_weights, repeats=P_size, axis=0).reshape(-1)  # this duplicate the self_point P_size times
        P_weights_repeat_all = np.repeat(P_weights.reshape(1, -1), repeats=self_size,axis=0).reshape(-1)  # this duplicate the P_points self_size times

        self_points_repeat_each_point_minus_P_points_repeat_all = np.sum((self_points_repeat_each_point - P_points_repeat_all)** 2, axis=1)
        all_distances_unreshaped = self_points_repeat_each_point_minus_P_points_repeat_all * self_weights_repeat_each_point * P_weights_repeat_all
        all_distances_reshaped = all_distances_unreshaped.reshape(-1, P_size)
        all_distances = np.min(all_distances_reshaped, axis=1)
        all_distances_is_nan = np.where(np.isnan(all_distances))

        if type == "by rate":
            m = int(m * self.get_size())  # number of points is m percents of n
        m_th_distance = np.partition(all_distances, m)[m]  # the m-th distance
        distances_smaller_than_median_indices = list(np.where(all_distances <= m_th_distance))  # all the m smallest distances indices in self.points
        P_subset = self_points[tuple(distances_smaller_than_median_indices)]
        w_subset = self_weights[tuple(distances_smaller_than_median_indices)]
        indexes_subset = self.indexes[tuple(distances_smaller_than_median_indices)]
        return SetOfPoints(P_subset, w_subset, indexes=indexes_subset)

    ##################################################################################

    def get_farthest_points_to_set_of_points(self, P, m, type):
        """
        Args:
            P (SetOfPoints) : a set of points
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"

        Returns:
            SetOfPoints: the points that are closest to the given set of points, by rate or by
                         fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        self_size = self.get_size()
        self_points = np.asarray(self.points)
        self_weights = np.asarray(self.weights)
        P_size = P.get_size()
        P_points = np.asarray(P.points)
        P_weights = np.asarray(P.weights)

        self_points_repeat_each_point = np.repeat(self_points, repeats=P_size, axis=0) #this duplicate the self_point P_size times
        P_points_repeat_all = np.repeat(P_points.reshape(1, -1), repeats=self_size, axis=0).reshape(-1, self.dim) #this duplicate the P_points self_size times

        self_weights_repeat_each_point = np.repeat(self_weights, repeats=P_size, axis=0).reshape(-1)  # this duplicate the self_point P_size times
        P_weights_repeat_all = np.repeat(P_weights.reshape(1, -1), repeats=self_size,axis=0).reshape(-1)  # this duplicate the P_points self_size times

        self_points_repeat_each_point_minus_P_points_repeat_all = np.sum((self_points_repeat_each_point - P_points_repeat_all)** 2, axis=1)
        all_distances_unreshaped = self_points_repeat_each_point_minus_P_points_repeat_all * self_weights_repeat_each_point * P_weights_repeat_all
        all_distances_reshaped = all_distances_unreshaped.reshape(-1, P_size)
        all_distances = np.min(all_distances_reshaped, axis=1)
        if type == "by rate":
            m = int(m * self.get_size())  # number of points is m percents of n
        partition_index = self_size - m
        m_th_distance = np.partition(all_distances, partition_index)[partition_index]  # the m-th distance
        distances_smaller_than_median_indices = list(np.where(all_distances >= m_th_distance))  # all the m smallest distances indices in self.points
        P_subset = self_points[distances_smaller_than_median_indices]
        w_subset = self_weights[distances_smaller_than_median_indices]
        indexes_subset = self.indexes[distances_smaller_than_median_indices]
        return SetOfPoints(P_subset, w_subset, indexes=indexes_subset)

    ##################################################################################

    def get_farthest_points_to_point(self, point, m, type):
        """
        Args:
            point (np.ndarray) : d-dimensional point
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"

        Returns:
            SetOfPoints: the points that are farthest to the given point, by rate or by
                         fixed number
        """
        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of points in query is larger than number of points in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) Number of points in query is larger than number of points in the set"

        size = self.get_size()

        #this calculates all the l2 squared distances from the given point to each point in the set
        point_repeat = np.repeat(point.reshape(1, -1), repeats=size, axis=0).reshape(-1, self.dim) #this duplicate the point n times where n is the number of points in set
        the_substract = point_repeat - self.points #substract each coordinate from its corresponding coordinate
        the_multiply = (np.multiply(the_substract, the_substract)) #squared of each one of the substract results
        the_plus = np.sum(the_multiply, axis=1) #sum all the squared substractions
        all_distances = np.multiply(self.weights, the_plus)#multiply all the distances by the weights of wach corresponding point in the set
        if type == "by rate":
            m = int(m * self.get_size()) #number of points is m percents of n
        m_th_distance = np.partition(all_distances, size - m)[size - m] #the m-th distance
        distances_smaller_than_median_indices = np.where(all_distances >= m_th_distance) #all the m largest distances indices in self.points
        P_subset = self.points[distances_smaller_than_median_indices]
        w_subset = self.weights[distances_smaller_than_median_indices]
        indexes_subset = self.indexes[distances_smaller_than_median_indices]
        return SetOfPoints(P_subset, w_subset, indexes=indexes_subset)


    ##################################################################################

    def get_sum_of_distances_to_point(self, point, weight=1):
        """
        Args:
            point (np.ndarray) : d-dimensional point
            weight (float, optional) : weight of point

        Returns:
            float: the sum of weighted distances to the given weighted point
        """

        assert weight > 0, "weight not positive"

        P = self.points
        point = point.reshape(-1, self.dim)
        w = self.weights.reshape(-1, 1)
        size = self.get_size()

        # this calculates the sum of all the l2 squared distances from the given point to each point in the set
        point_repeat = np.repeat(point.reshape(1, -1), repeats=size, axis=0).reshape(-1, self.dim) #this duplicate the point n times where n is the number of points in set
        the_substract = point_repeat - self.points #substract each coordinate from its corresponding coordinate
        the_multiply = (np.multiply(the_substract, the_substract)) #squared of each one of the substract results
        the_plus = np.sum(the_multiply, axis=1).reshape(-1,1) #sum all the squared substractions
        all_distances = np.multiply(w, the_plus) #multiply all the distances by the weights of wach corresponding point in the set
        sum_of_distances = sum(all_distances)
        return sum_of_distances

    ##################################################################################

    def get_median(self, sample_size_rate, closest_rate):
        """
        Args:
            sample_size_rate (float) : the size of the sample relative to the set
            closest_rate (float) : the size of closest points to the median relative to the set

        Returns:
            np.ndarray: the median of the set. See Alg. 3 in the paper;
        """

        assert sample_size_rate < 1 and sample_size_rate > 0, "sample_size_rate not in (0,1)"
        assert closest_rate < 1 and closest_rate > 0, "closest_rate not in (0,1)"
        assert self.get_size() != 0, "there is no points in the set is zero"

        size_of_sample = int(sample_size_rate*self.get_size())
        if size_of_sample == 0:
            S = self
        else:
            S = self.get_sample_of_points(size_of_sample)
        points = S.points
        weights = S.weights
        dim = self.dim
        size = len(points)

        points_each_repeat = np.repeat(points, repeats=size, axis=0) #this generates [[p_1, p_1, p_1,... <n times>],[p_2,p_2,p_2,... <n times>],...,[p_n,p_n,p_n,... <n times>]]
        points_all_repeat = np.repeat(points.reshape(1, -1), repeats=size, axis=0).reshape(-1, dim) #this generates [[p_1, p_2, p_3,...,p_n],[p_1, p_2, p_3,...,p_n],...<n times>]]
        weights_all_repeat = np.repeat(weights.reshape(-1), repeats=size).reshape(-1)
        the_substract = points_each_repeat - points_all_repeat #substract each coordinate from its corresponding coordinate
        the_multiply = (np.multiply(the_substract, the_substract)) #squared of each one of the substract results
        the_plus = np.sum(the_multiply, axis=1).reshape(-1, 1) #sum all the squared substractions
        all_distances = np.multiply(weights_all_repeat.reshape(-1, 1), the_plus) #multiply all the distances by the weights of wach corresponding point in the set
        all_distances_per_point = all_distances.reshape(size, -1) #that generates n \times n matrix, where the i,j entry is the weighted squared distance from the i-th points in the set to the j-th point int the set
        #print("all_distances_per_point: \n", all_distances_per_point)
        number_of_closest = int(size * closest_rate)
        all_distances_per_point_medians_in_the_middle = np.partition(all_distances_per_point, number_of_closest , axis=1) #this puts the median distance at each row in the middle
        #print("all_distances_per_point_medians_in_the_middle: \n", all_distances_per_point_medians_in_the_middle)
        all_cosest_distances_per_point = all_distances_per_point_medians_in_the_middle[:, 0:number_of_closest] #this removes all the entries that are right to the medians in all the rows. That is, n \times n/2 array, where the distances in row i is the distances to the closest n/2 points to the i-th point in the set
        sum_of_closest_distances_per_point = np.sum(all_cosest_distances_per_point, axis=1) #this generates 1 times n array of distances, where the i-th distance is the sum of weighted sqared dostances from the i-th point in the set to its n/2 closest points
        median_index = np.argmin(sum_of_closest_distances_per_point) #the index of the median
        median_point = points[median_index]
        return median_point

    ######################################################################

    def set_points(self, points):
        """
        :param points: numpy ndarray, sets all the points fo the set
        :return: None
        """
        assert self.get_size() > 0, "set is empty"
        self.points = points

    ######################################################################

    def set_all_sensitivities(self, sensitivity):
        """
        The method gets a number and set all the sensitivities to be that number
        Args:
            sensitivity (float) : the sensitivity we set for all the points in the set

        Returns:
            ~
        """

        assert sensitivity > 0, "sensitivity is not positive"
        assert self.get_size() > 0, "set is empty"

        new_sensitivities = np.ones((self.get_size(), 1), dtype=np.float) * sensitivity
        self.sensitivities = new_sensitivities

    ######################################################################

    def set_weights(self, T, m):
        """
        The method sets the weights in the set to as described in line 10 int the main alg;
        Args:
            T (float) : sum of sensitivities
            m (int) : coreset size

        Returns:
            ~
        """

        assert self.get_size() > 0, "set is empty"

        numerator = self.weights.reshape(-1,1) * T #np.ones((self.get_size(), 1), dtype=np.float) * T
        denominator = self.sensitivities * m
        new_weights = numerator / denominator

        self.weights = new_weights

    #######################################################################

    def get_probabbilites(self):
        """
        The method returns the probabilities to be choosen as described in line 9 in main alg
        Returns:
            np.ndarray: the probabilities to be choosen
        """

        T = self.get_sum_of_sensitivities()

        probs = self.sensitivities / T
        return probs

    #########################################################################

    def set_sensitivities(self, k):
        """
        The method set the sensitivities of the points in the set as decribed in line 5 in main alg.
        Args:
            k (int) : number of outliers

        Returns:
            ~
        """

        assert self.get_size() > 0, "set is empty"

        size = self.get_size()
        sensitivities1 = np.ones((self.get_size(), 1), dtype=np.float) * ((1*k)/size)

        sorted_indices = np.expand_dims(np.argsort(self.points[:, 0]).astype(np.float) + 1, axis=1)
        sensitivities2 = np.reciprocal(sorted_indices)
        sensitivities3 = np.reciprocal(np.subtract((self.get_size() + 1) * np.ones((self.get_size(), 1)), sorted_indices))
        self.sensitivities = np.maximum(sensitivities1, np.maximum(sensitivities2, sensitivities3))

    #########################################################################

    def get_arbitrary_sensitivity(self):
        """
        The method returns an arbitrary sensitivity from the set
        Returns:
            float: a random sensitivity from the set
        """

        assert self.get_size() > 0, "set is empty"

        num = random.randint(-1, self.get_size() - 1)
        return self.sensitivities[num]

    #########################################################################

    def get_robust_cost_to_point(self, point, k):
        """
        Args:
            point (np.ndarray) : d-dimensional point
            k (int) : number of outliers

        Returns:
            float:  the sum of weighted distances to the (size of set)-k closest points in the set the given point
        """

        closest = self.get_closest_points_to_point(point, self.get_size() - k, "by number")
        total_cost = closest.get_sum_of_distances_to_point(point)
        return total_cost

    ###########################################################################

    def get_cost_to_center_without_outliers(self, centers, outliers):
        """
        TODO: complete
        :param centers:
        :param outliers:
        :return:
        """

        outliers_size = outliers.get_size()
        self_size = self.get_size()

        #closest = self.get_closest_points_to_set_of_points(centers, self_size - outliers_size, "by number")
        #total_cost = closest.get_sum_of_distances_to_set_of_points(centers)
        Q = copy.deepcopy(self)
        Q.remove_from_set(outliers)
        total_cost = Q.get_sum_of_distances_to_set_of_points(centers)
        return total_cost

    ###########################################################################

    def get_points_at_indices(self, start, end):
        """
        Args:
            start (int) : starting index
            end (end) : ending index

        Returns:
            SetOfPoints: a set of point that contains the points in the given range of indices
        """

        size = end - start
        indices = np.asarray(range(size)) + start

        P_subset = self.points[indices]
        w_subset = self.weights[indices]
        sen_subset = self.sensitivities[indices]
        indexes_subset = self.indexes[indices]
        return SetOfPoints(P_subset, w_subset, sen_subset, indexes_subset)

    ###########################################################################

    def get_2_approx_points(self, k):
        """
        This function gets integer k>0 and returns k points that minimizes the sum of squared distances to the points
        in the set up to contant factor
        :param k:
        :return:
        """
        points = self.points
        weights = self.weights
        dim = self.dim
        size = len(points)

        points_each_repeat = np.repeat(points, repeats=size,axis=0)  # this generates [[p_1, p_1, p_1,... <n times>],[p_2,p_2,p_2,... <n times>],...,[p_n,p_n,p_n,... <n times>]]
        points_all_repeat = np.repeat(points.reshape(1, -1), repeats=size, axis=0).reshape(-1,dim)  # this generates [[p_1, p_2, p_3,...,p_n],[p_1, p_2, p_3,...,p_n],...<n times>]]
        weights_all_repeat = np.tile(weights, size).transpose().reshape(1,-1).transpose()  # this generates n^2 weights, that is n weights that coressponding for each one of the n duplications of the points in the set
        the_substract = points_each_repeat - points_all_repeat  # substract each coordinate from its corresponding coordinate
        the_multiply = (np.multiply(the_substract, the_substract))  # squared of each one of the substract results
        the_plus = np.sum(the_multiply, axis=1).reshape(-1, 1)  # sum all the squared substractions
        all_distances = np.multiply(weights_all_repeat,the_plus)  # multiply all the distances by the weights of wach corresponding point in the set
        all_distances_per_point = all_distances.reshape(size, -1)  # that generates n \times n matrix, where the i,j entry is the weighted squared distance from the i-th points in the set to the j-th point int the set
        all_distances_from_each_point_to_entire_set = np.sum(all_distances_per_point, axis=1)
        size = self.get_size()
        all_indices = np.asarray(range(size)).reshape(1, -1)
        all_indices_repeat = np.repeat(all_indices, k, axis=0)

        all_k_combination_of_all_indices = np.array(np.meshgrid(*all_indices_repeat)).T.reshape(-1, k)
        all_k_combination_of_all_indices_flat = np.asarray(all_k_combination_of_all_indices).reshape(-1)
        distance_from_points_to_all_set_by_all_combinations = all_distances_per_point[all_k_combination_of_all_indices_flat]
        distance_from_points_to_all_set_by_all_combinations_reshaped = distance_from_points_to_all_set_by_all_combinations.reshape(-1,k,size)
        all_distances_from_each_combination_to_all_set = np.min(distance_from_points_to_all_set_by_all_combinations_reshaped, axis=1)
        sum_of_distances_from_each_combination_to_all_set = np.sum(all_distances_from_each_combination_to_all_set, axis=1)
        min_index_of_combination = np.argmin(sum_of_distances_from_each_combination_to_all_set)
        min_indices = all_k_combination_of_all_indices[min_index_of_combination]
        min_centers = points[min_indices]
        x = 2
        return min_centers

    ###########################################################################

    def get_sum_of_distances_to_set_of_points(self, centers):
        """
        This function gets a set of k centers and returns the sum of squared distances to these centers
        :param centers:
        :return:
        """

        self_points = self.points
        centers_points = centers.points
        self_weights = self.weights
        centers_weights = centers.weights
        dim = self.dim
        size = len(self_points)
        k = centers.get_size()

        centers_points_each_repeat = np.repeat(centers_points, repeats=size, axis=0)
        centers_weights_each_repeat = np.repeat(centers_weights, repeats=size, axis=0).reshape(-1)
        self_points_all_repeat = np.repeat(self_points.reshape(1, -1), repeats=k, axis=0).reshape(-1, dim) #this generates [[p_1, p_2, p_3,...,p_n],[p_1, p_2, p_3,...,p_n],...<n times>]]
        self_weights_all_repeat = np.repeat(self_weights.reshape(1, -1), repeats=k, axis=0).reshape(-1)
        centers_points_each_repeat_minus_self_points_all_repeat = centers_points_each_repeat - self_points_all_repeat
        squared_norms = np.sum(np.multiply(centers_points_each_repeat_minus_self_points_all_repeat, centers_points_each_repeat_minus_self_points_all_repeat),axis=1)
        all_distances = squared_norms * centers_weights_each_repeat * self_weights_all_repeat
        all_distances_from_each_center = all_distances.reshape(-1,size)
        min_distances = np.min(all_distances_from_each_center, axis=0)
        cost = np.sum(min_distances)
        return cost

    ###########################################################################

    def get_number_of_points_larger_than_value(self, value):
        """
        TODO: complete
        :param value:
        :return:
        """
        assert self.dim == 1, "dimension not fit to this task, only works when d=1"

        counter = 0
        the_points = []
        for point in self.points:
            if point > value:
                the_points.append(point)
                counter += 1
        return counter

    ###########################################################################

    def sort_by_indexes(self):
        self_size = self.get_size()
        self_points = self.points
        self_weights = self.weights
        self_sensitivities = self.sensitivities
        self_indexes = self.indexes
        new_points = []
        new_weights = []
        new_sensitivities = []
        new_indexes = []
        for i in range(self_size):
            for j in range(self_size):
                if self.indexes[j] == i:
                    new_points.append(self_points[j])
                    new_weights.append(self_weights[j])
                    new_sensitivities.append(self_sensitivities[j])
                    new_indexes.append(self_indexes[j])
        self.points = np.asarray(new_points)
        self.weights = np.asarray(new_weights)
        self.sensitivities = np.asarray(new_sensitivities)
        self.indexes = np.asarray(new_indexes)

    #########################################################

# from here it's my code, don't change the code above!
    def copy_array(self, array):
        return array.copy()

    #########################################################

    def get_copy(self):
        p, w, sen, indexes = self.copy_array(self.points), self.copy_array(self.weights), self.copy_array(self.sensitivities),\
                             self.copy_array(self.indexes)
        return SetOfPoints(p, w, sen, indexes)

    #########################################################

    def get_line_between_two_points(self, point1, point2):
        if point1[0] == point2[0]:
            slope = sys.maxsize
        elif point1[1] == point2[1]:
            slope = sys.maxsize
        else:
            slope = (point2[1] - point1[1]) / (point2[0] - point1[0])
        y_intercept = point1[1] - slope * point1[0]
        return Line(point1, point2, slope, y_intercept)

    #########################################################

    def get_line_between_every_two_points(self):
        lines = []
        for i in range(self.get_size()):
            for j in range(i + 1, self.get_size()):
                lines.append(self.get_line_between_two_points(self.points[i], self.points[j]))
        return lines

    #########################################################

    def sorted_indices_by_cost(self, lines):
        costs = []
        for point in self.points:
            cost_point = sys.maxsize
            for line in lines:
                dist = line.distance_from_point(point)
                cost_point = dist if dist < cost_point else cost_point
            costs.append(cost_point)
        return costs

    #########################################################

    def get_set_of_points_with_indices(self, indices):
        return SetOfPoints(self.points[indices],  self.weights[indices], self.sensitivities[indices], self.indexes[indices])

    #########################################################

    def remove_points(self, sorted_indeces_by_costs, percentage_of_points_to_reduce_every_iteration):
        """
        :param sorted_indeces_by_costs: costs of points from the Bi-Criteria
        :param percentage_of_points_to_reduce_every_iteration: percentage of points to remove every iteration
        :return: (SetofPoints) the set of points without the points that were with the smallest cost
        """
        indices_to_remove = sorted_indeces_by_costs[:int(percentage_of_points_to_reduce_every_iteration * self.get_size())]
        set_of_points_to_remove = self.get_set_of_points_with_indices(indices_to_remove)
        self.remove_from_set(set_of_points_to_remove)

    #########################################################

    def project_to_lines(self, lines, plot=False):
        points_on_Bi_Criteria = {}
        for line in lines:
            points_on_Bi_Criteria[line] = SetOfPoints()
        for point, i in zip(self.points, self.indexes):
            min_ = sys.maxsize
            closest_line = None
            for line in lines:
                if line.distance_from_point(point) < min_:
                    min_ = line.distance_from_point(point)
                    closest_line = line

            projected_point = point.copy()
            projected_point = closest_line.project_point_to_line(projected_point)
            points_on_Bi_Criteria[closest_line].add_set_of_points(SetOfPoints(np.array([projected_point]), np.ones(1),
                                                                              np.ones(1), np.array(i)))


        if plot:
            plt.figure()
            colors = []
            for i in range(len(lines)):
                colors.append((random.random(), random.random(), random.random()))
            for i, group in enumerate(points_on_Bi_Criteria.values()):
                plt.scatter(group.points[:, 0], group.points[:, 1], c=colors[i])
            plt.show()
        return points_on_Bi_Criteria

    #########################################################
    def coreset_return_sensitivities(self, P, k, m):
        """
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            m (int) : size of wanted coreset
        Returns:
            SetOfPoints: the coreset of P for k weighted centers. See Alg. 2 in the paper;
        """
        median_sample_size = self.parameters_config.median_sample_size
        closest_to_median_rate = self.parameters_config.closest_to_median_rate
        assert k > 0, "k is not a positive integer"
        assert m > 0, "m is not a positive integer"
        assert P.get_size() != 0, "Q size is zero"
        number_of_remains_multiply_factor = self.parameters_config.number_of_remains_multiply_factor
        max_sensitivity_multiply_factor = self.parameters_config.max_sensitivity_multiply_factor
        minimum_number_of_points_in_iteration = k*number_of_remains_multiply_factor #int(math.log(P.get_size()))
        Q = copy.deepcopy(P)
        temp_set = SetOfPoints()
        max_sensitivity = -1
        flag1 = False
        flag2 = False
        while True:
            [q_k, Q_k] = self.recursive_robust_median(Q, k, median_sample_size, closest_to_median_rate) #get the recursive median q_k and its closest points Q_k
            if Q_k.get_size() == 0:
                flag1 = True
                continue
            Q_k.set_sensitivities(k)# sets all the sensitivities in Q_k as described in line 5 in main alg.
            current_sensitivity = Q_k.get_arbitrary_sensitivity()
            if current_sensitivity > max_sensitivity:
                max_sensitivity = current_sensitivity #we save the maximum sensitivity in order to give the highest sensitivity to the points that remains in Q after this loop ends
            temp_set.add_set_of_points(Q_k) #since we remove Q_k from Q each time, we still want to save every thing in order to go over the entire points after this loop ends and select from them and etc., so we save everything in temp_set
            Q.remove_from_set(Q_k)
            size = Q.get_size()
            size_Q_k = Q_k.get_size()
            Q_k_weigted_size = Q_k.get_sum_of_weights()
            if size <= minimum_number_of_points_in_iteration or Q_k_weigted_size == 0: # stop conditions
                flag2 = True
                break
        if Q.get_size() > 0:
            Q.set_all_sensitivities(max_sensitivity * max_sensitivity_multiply_factor) # here we set the sensitivities of the points who left to the highest - since they are outliers with a very high probability
            temp_set.add_set_of_points(Q) #and now temp_set is all the points we began with - just with updated sensitivities
        T = temp_set.get_sum_of_sensitivities()
        temp_set.set_weights(T, m) #sets the weights as described in line 10 in main alg
        #temp_set.sort_by_indexes() #change this line in tests for sensativities bound
        return temp_set#temp_set.sensitivities, temp_set.weights

    #########################################################
    def computeSensativities(self):
        con1 = np.vectorize(self.f1)
        con2 = np.vectorize(self.f2)
        cluster_WKmeans = self.coreset_return_sensitivities(self, self.parameters_config.k,
                                                         self.parameters_config.coreset_size)
        idx_sort = self.sortOnSubspace()

        S = np.maximum.reduce([con1(idx_sort), con2(idx_sort, self.get_size()), np.squeeze(
            cluster_WKmeans.sensitivities) if cluster_WKmeans.get_size() > 1 else cluster_WKmeans.sensitivities])
        # plt.scatter(self.points[:, 0], self.points[:, 1])
        # plt.show()
        return S


    #########################################################

    @staticmethod
    def f1(x):
        return 1 / x

    #########################################################
    @staticmethod
    def f2(x, n):
        return 1 / (n - x + 1)

    #########################################################
    def sortOnSubspace(self):
        if self.get_size() == 1:
            return np.array([1])
        else:
            v = self.points[0] - self.points[1]
            I = np.apply_along_axis(lambda x: np.dot(x, v), axis=1, arr=self.points)
            y = np.zeros_like(I)
            y[np.argsort(I)] = np.arange(I.shape[0]) + 1
            return y

    #########################################################
    def recursive_robust_median(self, P, k, median_sample_size, recursive_median_closest_to_median_rate):
        """
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            median_sample_size (int) : the size of closest points to the median
            recursive_median_closest_to_median_rate (float) : parameter for the median

        Returns:
            [np.ndarray, SetOfPoints]: the recursive robust median of P and its closest points. See Alg. 1 in the paper;
        """

        iterations_median = self.parameters_config.iterations_median

        assert k > 0, "k is not a positive integer"
        assert recursive_median_closest_to_median_rate < 1 and recursive_median_closest_to_median_rate > 0, "closest_rate2 not in (0,1)"
        assert P.get_size() != 0, "Q size is zero"

        minimum_number_of_points_in_iteration = int(np.log(P.get_size())) #for stop condition
        Q = copy.deepcopy(P)
        q = []
        for i in range(iterations_median):#used to be until k
            size_of_sample = median_sample_size
            q = Q.get_sample_of_points(size_of_sample)
            Q = Q.get_closest_points_to_set_of_points(q, recursive_median_closest_to_median_rate, type="by rate") #the median closest points

            size = Q.get_size()
            if size <= minimum_number_of_points_in_iteration:
                break
        return [q, Q]