from pybrain.optimization import HillClimber
from NeuralNetLearner import NeuralNetLearner

class HillClimbingOptimizer():
    def __init__(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()
        self.optimizer = HillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                                     maxEvaluations = 3, verbose = True, numParameters = 661)

    def learn(self):
        self.neural_net = self.optimizer.learn()

h = HillClimbingOptimizer()
h.learn()