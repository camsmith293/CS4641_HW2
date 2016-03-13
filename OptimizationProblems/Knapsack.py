from itertools import combinations
from random import choice, randint
from pybrain.structure.evolvables.evolvable import Evolvable

class Knapsack(Evolvable):

    def __init__(self):
        self.items = [
            ("map", 9, 150), ("compass", 13, 35), ("water", 153, 200), ("sandwich", 50, 160),
            ("glucose", 15, 60), ("tin", 68, 45), ("banana", 27, 60), ("apple", 39, 40),
            ("cheese", 23, 30), ("beer", 52, 10), ("suntan cream", 11, 70), ("camera", 32, 30),
            ("t-shirt", 24, 15), ("trousers", 48, 10), ("umbrella", 73, 40),
            ("waterproof trousers", 42, 70), ("waterproof overclothes", 43, 75),
            ("note-case", 22, 80), ("sunglasses", 7, 20), ("towel", 18, 12),
            ("socks", 4, 50), ("book", 30, 10),
        ]

        self.model = [0] * len(self.items)
        self.randomize()

    def f(self):
        ' Totalise a particular combination of items'
        totwt = totval = 0
        for item in range(len(self.model)):
            wt = self.items[item][1]
            val = self.items[item][2]
            if self.model[item]:
                totwt  += wt
                totval += val
        return totval if totwt <= 400 else 0

    def mutate(self, **args):
        index = choice(list(range(0,len(self.model))))
        if self.model[index] is 0:
            self.model[index] = 1
        else:
            self.model[index] = 0

    def randomize(self):
        for i in range(len(self.model)):
            self.model[i] = choice([0,1])

    def domain(self):
        return [0,1] * len(self.model)