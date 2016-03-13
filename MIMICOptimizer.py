from pybrain.optimization.distributionbased import distributionbased
from copy import deepcopy
from NeuralNetLearner import NeuralNetLearner
from Mimic import Mimic

class MIMICOptimizer():

    def __init__(self, to_minimize):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()

        self.optimizer = Mimic([(-100, 100)] * 661, self.NeuralNet_fitness, samples=250)

    def learn(self, iterations):
        for i in range(iterations):
            print("Iteration ", i)
            self.optimizer.fit()

    def NeuralNet_fitness(self, weights):
        evaluatee = deepcopy(self.neural_net)
        evaluatee._setParameters(weights)
        return self.testing_set.evaluateModuleMSE(evaluatee)

m = MIMICOptimizer(True)
m.learn(100)