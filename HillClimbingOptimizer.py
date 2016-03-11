from pybrain.optimization import HillClimber
from NeuralNetLearner import NeuralNetLearner

class HillClimbingOptimizer():
    def __init__(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set = self.learner.train_data
        self.testing_set = self.learner.test_data
        self.optimizer = HillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True)

    def learn(self,n_steps):
        for i in range(n_steps):
            print("Iteration %d:", i)
            self.neural_net = self.optimizer.learn()
            print(self.testing_set.evaluateModuleMSE(self.neural_net), "\n")

h = HillClimbingOptimizer()
h.learn(1)