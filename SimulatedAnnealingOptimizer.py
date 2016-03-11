from pybrain.optimization import StochasticHillClimber
from NeuralNetLearner import NeuralNetLearner

class SimulatedAnnealingOptimizer():
    def __init__(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()
        self.optimizer = StochasticHillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                                     verbose = True, numParameters = 661)

    def learn(self):
        self.neural_net = self.optimizer.learn()

s = SimulatedAnnealingOptimizer()
s.learn()