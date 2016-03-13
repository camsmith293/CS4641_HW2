from copy import deepcopy

from pybrain.optimization.populationbased import GA
from NeuralNetLearner import NeuralNetLearner

from OptimizationProblems.FourPeaks import FourPeaks, fitness_fourpeaks_GA
from OptimizationProblems.kColors import kColors, fitness_kcolors_GA
from OptimizationProblems.Knapsack import Knapsack, fitness_knapsack_GA

class GeneticAlgorithmOptimizer():
    def learn_nnet(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()
        self.optimizer = GA(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                            verbose = True, numParameters = 661,
                            maxLearningSteps=2000, desiredEvaluation = 0.6)
        temp, best_estimate = self.optimizer.learn()
        return temp

    def learn_optimizationproblem(self, problem, fitness_function):
        initial_population = [problem.model]
        for i in range(24):
            initial_population.append(deepcopy(problem).randomize())
        self.optimizer = GA(fitness_function, problem.model,
                            verbose = True,maxLearningSteps=2000, desiredEvaluation = 0.6,
                            initialPopulation = initial_population, initRangeScaling=1)
        temp, best_estimate = self.optimizer.learn()
        return temp

g = GeneticAlgorithmOptimizer()
k = Knapsack()
g.learn_optimizationproblem(k, fitness_knapsack_GA)