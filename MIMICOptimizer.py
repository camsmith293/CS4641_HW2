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

        self.optimizer = Mimic([(-10, 10)] * 661, self.NeuralNet_fitness, samples=500, maximize=False)

        evaluations = []
        nnet_mimic_evaluations_file = open('out/nnet_mimic_evaluations.csv', 'a')

        for i in range(iterations):
            print("Iteration %d" % i)
            filtered = self.optimizer.fit()
            evaluations.append(filtered[1])
            print(filtered)
            if (filtered[0][0] is filtered[0][-1]):
                for item in evaluations:
                    nnet_mimic_evaluations_file.write("%s\n" % item)
                evaluations.append("end")
                return filtered[0][0], filtered[1]

    def NeuralNet_fitness(self, weights):
        self.neural_net._setParameters(weights)
        return self.testing_set.evaluateModuleMSE(self.neural_net)

    def learn_optimizationproblem(self, iterations, problem, fitness_function):
        self.optimizer = Mimic(problem.domain(), fitness_function, samples=250,
                maximize=True, discreteValues=True, percentile=0.5)

        evaluations = []
        out_name = 'out/opt_mimic_evaluations_' + problem.__class__.__name__ + '.csv'
        opt_mimic_evaluations_file = open(out_name, 'a')

        for i in range(iterations):
            print("Iteration %d" % i)
            filtered = self.optimizer.fit()
            evaluations.append(filtered[1])
            print(filtered)
            if (filtered[0][0] is filtered[0][-1]):
                for item in evaluations:
                    opt_mimic_evaluations_file.write("%s\n" % item)
                evaluations.append("end")
                return filtered[0][0], filtered[1]

m = MIMICOptimizer()
# f = FourPeaks('11110000')
# m.learn_optimizationproblem(1000, f, fitness_fourpeaks)
m.learn_nnet(10)