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
f_assigned = {'Black', 'Black', 'Black', 'Black',}

h.learn_optimizationproblem(2, f, fitness_kcolors, minimize=True)
s.learn_optimizationproblem(2, f, fitness_kcolors, minimize=True)
g.learn_optimizationproblem(f, fitness_kcolors_GA, minimize=True)
m.learn_optimizationproblem(500, f, fitness_kcolors_GA, minimize=True)