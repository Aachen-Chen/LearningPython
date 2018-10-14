import numpy as np


class NumPyExample():
    @staticmethod
    def calculation():
        # ndarray allow +-*/ directly
        print(np.ones((5, 2)) * 2)

    @staticmethod
    def shaping():
        print(np.c_[np.ones((5, 1)), np.ones((5, 2))])

        sequence24 = np.arange(8).reshape(2, 4)
        sequence42 = sequence24.reshape(4, 2)
        print(sequence24)
        print(sequence42)

        sequence_ = sequence42.reshape(-1, 1)
        print(sequence_)

        sequence_ = sequence42.reshape(1, -1)
        print(sequence_)


NumPyExample.shaping()



