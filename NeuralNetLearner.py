import numpy as np
from scipy.ndimage import convolve
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from sklearn.datasets import load_digits

def nudge_dataset(X, Y):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """
        direction_vectors = [
            [[0, 1, 0],
             [0, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [1, 0, 0],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 1],
             [0, 0, 0]],

            [[0, 0, 0],
             [0, 0, 0],
             [0, 1, 0]]]

        shift = lambda x, w: convolve(x.reshape((8, 8)), mode='constant',
                                      weights=w).ravel()
        X = np.concatenate([X] +
                           [np.apply_along_axis(shift, 1, X, vector)
                            for vector in direction_vectors])
        Y = np.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

class NeuralNetLearner:
    def __init__(self):
        self.bunch = load_digits()
        self.X = np.asarray(self.bunch.data, 'float32')
        self.X, self.Y = nudge_dataset(self.X, self.bunch.target)
        self.X = (self.X - np.min(self.X, 0)) / (np.max(self.X, 0) + 0.0001)  # 0-1 scaling

        self.ds = ClassificationDataSet(64, nb_classes=10, class_labels=self.bunch.target_names)
        for (x, y) in zip(self.X, self.Y):
            self.ds.addSample(x, y)

        self.test_data, self.train_data = self.ds.splitWithProportion(0.3)

        self.network = buildNetwork(64, 10, 1)

    def get_datasets(self):
        return self.train_data, self.test_data

    def activate(self, x):
        self.network.activate(x.tolist())

    def fitness_func(self, x):
        if not (x.size == 64):
            print("Bad input vector: ", x)
            return
        sum_of_squared_error = 0
        for (input, target) in self.ds:
            sum_of_squared_error += (target - self.activate(input.tolist()))
        return (sum_of_squared_error / self.ds.length)

    def get_weights(self):
        return
