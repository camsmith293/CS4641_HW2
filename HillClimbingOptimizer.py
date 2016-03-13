from pybrain.optimization import HillClimber
from NeuralNetLearner import NeuralNetLearner
from OptimizationProblems import *
import matplotlib.pyplot as plt
from OptimizationProblems.FourPeaks import FourPeaks
from OptimizationProblems.kColors import kColors
from OptimizationProblems.Knapsack import Knapsack


class HillClimbingOptimizer():

    def learn_nnet(self, num_restarts):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()

        # Optimizer will take 2000 steps and restart, saving the best model from the restarts
        self.optimizer = HillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                                     verbose = True, numParameters = 661, maxLearningSteps = 2000)

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
        return best_model

    def learn_optimizationproblem(self, num_restarts, problem):
        # Optimizer will take 250 steps and restart, saving the best model from the restarts
        self.optimizer = HillClimber(problem.f, problem, verbose = True, maxLearningSteps = 250)

        best_model = problem
        max_fitness = -2147438640

        temp = problem
        for i in range(num_restarts):
            temp = self.optimizer.learn()
            temp_fitness = temp.f()
            if temp_fitness >= best_model.f():
                best_model = temp
                max_fitness = temp_fitness

        return best_model

h = HillClimbingOptimizer()
f = FourPeaks()
h.learn_optimizationproblem(5, f)