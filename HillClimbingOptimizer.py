from pybrain.optimization import HillClimber
from NeuralNetLearner import NeuralNetLearner
from OptimizationProblems import *
from OptimizationProblems.FourPeaks import FourPeaks, fitness_fourpeaks
from OptimizationProblems.kColors import kColors, fitness_kcolors
from OptimizationProblems.Knapsack import Knapsack, fitness_knapsack


class HillClimbingOptimizer():

    def learn_nnet(self, num_restarts):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()

        # Optimizer will take 2000 steps and restart, saving the best model from the restarts
        self.optimizer = HillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                verbose = True, numParameters = 661, maxLearningSteps = 2000,  storeAllEvaluations = True)

        # Save best model and lowest MSE for random restarting
        best_model = self.neural_net
        min_MSE = 2147438647

        for i in range(num_restarts):
            temp, best_estimate = self.optimizer.learn()

            nnet_hc_evaluations_file = open('out/nnet_hc_evaluations.csv', 'a')
            for item in self.optimizer._allEvaluations:
                nnet_hc_evaluations_file.write("%s\n" % item)
            nnet_hc_evaluations_file.write("Restart %d\n" % i)

            self.optimizer = HillClimber(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                    verbose = True, numParameters = 661, maxLearningSteps = 1000,  storeAllEvaluations = True)
            if best_estimate <= min_MSE:
                best_model = temp
                min_MSE = best_estimate

        self.neural_net = best_model

        return best_model

    def learn_optimizationproblem(self, num_restarts, problem, fitness_function, minimize=False):
        # Optimizer will take 250 steps and restart, saving the best model from the restarts
        self.optimizer = HillClimber(fitness_function, problem, verbose = True,
                maxLearningSteps = 250, minimize=minimize, storeAllEvaluations = True)
        best_model = problem
        max_fitness = -2147438640

        for i in range(num_restarts):
            print("Restart", i)
            temp, best_estimate = self.optimizer.learn()

            out_name = 'out/opt_hc_evaluations_' + problem.__class__.__name__ + '.csv'
            opt_hc_evaluations_file = open(out_name, 'a')
            for item in self.optimizer._allEvaluations:
                opt_hc_evaluations_file.write("%s\n" % item)
            opt_hc_evaluations_file.write("Restart %d\n" % i)

            self.optimizer = HillClimber(fitness_function, problem, verbose = True,
                     maxLearningSteps = 250, minimize=True, storeAllEvaluations = True)
            if best_estimate >= max_fitness:
                best_model = temp
                max_fitness = best_estimate

        return best_model