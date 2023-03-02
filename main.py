import numpy as np
from set_of_points import SetOfPoints
from tests import Test
from parameters_config import ParameterConfig
from create_data import polygon_data
import pickle
def main():
    params = ParameterConfig()
    # create data
    # data = polygon_data(params.k, is_debug=True)
    # with open('data.p', 'wb') as handle:
    #     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data.p', 'rb') as handle:
        data = pickle.load(handle)
    P = SetOfPoints(data, parameters_config=params)
    sizes = np.linspace(20, 400, 15, dtype=int)
    test = Test(P, 5, sizes)
    test.testUniformSampling(plot=True)




if __name__ == "__main__":
    main()