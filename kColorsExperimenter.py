from OptimizationProblems.kColors import kColors, fitness_kcolors, fitness_kcolors_GA
from HillClimbingOptimizer import HillClimbingOptimizer
from SimulatedAnnealingOptimizer import SimulatedAnnealingOptimizer
from GeneticAlgorithmOptimizer import GeneticAlgorithmOptimizer
from MIMICOptimizer import MIMICOptimizer

h = HillClimbingOptimizer()
s = SimulatedAnnealingOptimizer()
g = GeneticAlgorithmOptimizer()
m = MIMICOptimizer()

f = kColors()

h.learn_optimizationproblem(2, f, fitness_kcolors)
s.learn_optimizationproblem(2, f, fitness_kcolors)
g.learn_optimizationproblem(f, fitness_kcolors_GA)
m.learn_optimizationproblem(500, f, fitness_kcolors_GA)