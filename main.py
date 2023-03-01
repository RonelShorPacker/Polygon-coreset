import numpy as np
from set_of_points import SetOfPoints
from tests import Test
from parameters_config import ParameterConfig
from create_data import polygon_data

def main():
    params = ParameterConfig()
    # create data
    data = polygon_data(params.k, is_debug=True)
    P = SetOfPoints(data, parameters_config=params)
    sizes = np.linspace(20, 200, 10, dtype=int)
    test = Test(P, 10, sizes)
    test.testUniformSampling()




if __name__ == "__main__":
    main()