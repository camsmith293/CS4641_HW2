from pybrain.optimization import HillClimber
from NeuralNetLearner import NeuralNetLearner

class HillClimbingOptimizer():
    def __init__(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        dataset = self.learner.ds
        self.optimizer = HillClimber(dataset.evaluateModuleMSE, self.neural_net, minimize=True)

    def learn(self,n_steps):
        for i in range(n_steps):
            self.neural_net = self.optimizer.learn()

h = HillClimbingOptimizer()
h.learn(1)