from OptimizationProblems.FourPeaks import FourPeaks, fitness_fourpeaks, fitness_fourpeaks_GA
from HillClimbingOptimizer import HillClimbingOptimizer
from SimulatedAnnealingOptimizer import SimulatedAnnealingOptimizer
from GeneticAlgorithmOptimizer import GeneticAlgorithmOptimizer
from MIMICOptimizer import MIMICOptimizer

h = HillClimbingOptimizer()
s = SimulatedAnnealingOptimizer()
g = GeneticAlgorithmOptimizer()
m = MIMICOptimizer()

f = FourPeaks('00001111')

h.learn_optimizationproblem(2, f, fitness_fourpeaks)
s.learn_optimizationproblem(2, f, fitness_fourpeaks)
g.learn_optimizationproblem(f, fitness_fourpeaks_GA)
m.learn_optimizationproblem(500, f, fitness_fourpeaks_GA)