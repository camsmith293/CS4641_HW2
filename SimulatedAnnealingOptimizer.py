from pybrain.optimization import StochasticHillClimber
from NeuralNetLearner import NeuralNetLearner

class SimulatedAnnealingOptimizer():
    def __init__(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()
        self.optimizer = StochasticHillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                                     verbose = True, numParameters = 661, maxLearningSteps = 2000)

    def learn_nnet(self, num_restarts):
         # Save best model and lowest MSE for random restarting
        best_model = self.neural_net
        min_MSE = 2147438647

        temp = self.neural_net
        for i in range(num_restarts):
            temp = self.optimizer.learn()
            temp_MSE = self.testing_set.evaluateModuleMSE(temp)
            if temp_MSE <= self.testing_set.evaluateModuleMSE(best_model):
                best_model = temp
                min_MSE = temp_MSE

        self.neural_net = best_model

s = SimulatedAnnealingOptimizer()
s.learn_nnet(5)