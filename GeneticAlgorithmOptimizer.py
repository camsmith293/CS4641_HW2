from pybrain.optimization.populationbased import GA
from NeuralNetLearner import NeuralNetLearner

class GeneticAlgorithmOptimizer():
    def learn_nnet(self):
        self.learner = NeuralNetLearner()
        self.neural_net = self.learner.network
        self.dataset = self.learner.ds
        self.training_set, self.testing_set = self.learner.get_datasets()
        self.optimizer = GA(self.training_set.evaluateModuleMSE, self.neural_net, minimize=True,
                            verbose = True, numParameters = 661,
                            maxLearningSteps=2000, desiredEvaluation = 0.6)
        self.neural_net = self.optimizer.learn()

    def learn_optimizationproblem(self, problem):
        self.optimizer = GA(problem.f, problem,
                             verbose = True, numParameters = 661,
                            maxLearningSteps=2000, desiredEvaluation = 0.6)
        self.neural_net = self.optimizer.learn()

g = GeneticAlgorithmOptimizer()
g.learn_nnet()