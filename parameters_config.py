import numpy as np
class ParameterConfig:
    def __init__(self):
        # main parameters
        self.header_indexes = None
        self.dim = 2
        self.lines_number = 4
        self.coreset_size = None

        # experiment  parameters
        self.sample_sizes = None
        self.inner_iterations = None
        self.centers_number = None
        self.outliers_trashold_value = None

        # EM k means for lines estimator parameters
        self.multiplications_of_k = None
        self.EM_iteration_test_multiplications = None

        # EM k means for points estimator parameters
        self.ground_true_iterations_number_ractor = None

        # k means for lines coreset parameters
        self.inner_a_b_approx_iterations = None
        self.sample_rate_for_a_b_approx = None

        # rectangles coreset - see paper
        self.minimum_points = 6
        self.sample_size = 7
        self.core_set_size = 100
        self.alpha = 2
        self.rho = 2
        self.gamma = 0.5
        self.lam = 1
        #distance function
        self.est = np.infty       
        # weighted centers coreset parameters
        self.median_sample_size = 20
        self.closest_to_median_rate = 0.5
        self.number_of_remains_multiply_factor = 0.5
        self.max_sensitivity_multiply_factor = 1

        # iterations
        self.RANSAC_iterations = None
        self.coreset_iterations = None
        self.RANSAC_EM_ITERATIONS = None
        self.coreset_to_ransac_time_rate = None
        self.iterations_median = 4
        self.exhaustive_search = 1000

        # files
        self.input_points_file_name = None

        #missing entries parameters
        self.missing_entries_alg = None
        self.cost_type= None
        self.KNN_k = None


        #data handler parameters
        self.points_number = None
        self.output_file_name = None

        # Bi-criteria parameters
        self.sample_size_bi_criteria = self.lines_number * 2 # k(j+1)
        self.reduce_each_iteration = 0.5
        self.fast_bi_criteria = True