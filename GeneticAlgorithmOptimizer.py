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
        self.optimizer = GA(self.testing_set.evaluateModuleMSE, self.neural_net, minimize=True,
                            verbose = True, numParameters = 661,
                            maxLearningSteps=1500, desiredEvaluation = 0.6,
                            storeAllEvaluations = True)
        temp, best_estimate = self.optimizer.learn()
        nnet_ga_evaluations_file = open('out/nnet_ga_evaluations.csv', 'a')
        for item in self.optimizer._allEvaluations:
            nnet_ga_evaluations_file.write("%s\n" % item)
        return temp

    def learn_optimizationproblem(self, problem, fitness_function, minimize=False):
        initial_population = [problem.model]
        for i in range(24):
            initial_population.append(deepcopy(problem).randomize())
        self.optimizer = GA(fitness_function, problem.model,
                            verbose = True,maxLearningSteps=2000, desiredEvaluation = 0.6, minimize=minimize,
                            initialPopulation = initial_population, initRangeScaling=1,storeAllEvaluations = True)
        out_name = 'out/opt_ga_evaluations_' + problem.__class__.__name__ + '.csv'
        opt_ga_evaluations_file = open(out_name, 'a')
        for item in self.optimizer._allEvaluations:
            opt_ga_evaluations_file.write("%s\n" % item)
        temp, best_estimate = self.optimizer.learn()
        return temp