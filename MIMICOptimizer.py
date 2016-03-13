from copy import deepcopy
from NeuralNetLearner import NeuralNetLearner
from Mimic import Mimic

from OptimizationProblems.FourPeaks import FourPeaks, fitness_fourpeaks
from OptimizationProblems.kColors import kColors, fitness_kcolors
from OptimizationProblems.Knapsack import Knapsack, fitness_knapsack

class MIMICOptimizer():
    def learn_nnet(self, iterations):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()

        self.optimizer = Mimic([(-1000, 1000)] * 661, self.NeuralNet_fitness, maximize=False)

        evaluations = []

        for i in range(iterations):
            print("Iteration %d" % i)
            filtered = self.optimizer.fit()
            evaluations.append(filtered[1])
            print(filtered)
            #print(len(set(filtered[0])))
            if len(filtered[0]) == 1:
                return filtered

    def NeuralNet_fitness(self, weights):
        evaluatee = deepcopy(self.neural_net)
        evaluatee._setParameters(weights)
        return self.testing_set.evaluateModuleMSE(evaluatee)

    def learn_optimizationproblem(self, iterations, problem, fitness_function):
        self.optimizer = Mimic(problem.domain(), fitness_function, samples=250,
                maximize=True, discreteValues=True, percentile=0.5)

        for i in range(iterations):
            print("Iteration %d" % i)
            filtered = self.optimizer.fit()
            print(filtered)
            if len(filtered) == 1:
                return filtered

m = MIMICOptimizer()
# f = FourPeaks('11110000')
# m.learn_optimizationproblem(1000, f, fitness_fourpeaks)
m.learn_nnet(1000)