from pybrain.optimization.distributionbased import distributionbased
from NeuralNetLearner import NeuralNetLearner
from Mimic import Mimic

class MIMICOptimizer():

    def __init__(self, to_minimize):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()

        self.optimizer = Mimic([(-2147483648, 2147483647)] * 661, self.training_set.evaluateModuleMSE)

    def learn(self, iterations):
        for i in range(iterations):
            print("Iteration ", i)
            self.optimizer.fit()

m = MIMICOptimizer()
m.learn(100)

