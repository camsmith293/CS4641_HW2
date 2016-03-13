from random import choice, randint
from pybrain.structure.evolvables.evolvable import Evolvable

def fitness_fourpeaks(evaluable):
        tail = evaluable.tail('0')
        head = evaluable.head('1')
        return max(tail, head) + evaluable.R()

def fitness_fourpeaks_GA(evaluable):
        stringed = ""
        for i in evaluable:
            stringed += str(evaluable[i])
        temp = FourPeaks(stringed)
        return fitness_fourpeaks(temp)

class FourPeaks(Evolvable):
    def __init__(self, bitstring):
        self.length = len(bitstring)
        self.t = self.length/10
        self.model = [0] * self.length
        for i in range(len(bitstring)):
            self.model[i] = int(bitstring[i])

    def tail(self, b):
        stringed = ""
        for i in self.model:
            stringed += str(self.model[i])
        return (len(stringed) - len(stringed.rstrip(b)))

    def head(self, b):
        stringed = ""
        for i in self.model:
            stringed += str(self.model[i])
        return len(stringed.split('0', 1)[0])

    def R(self):
        tail_bool = self.tail('0') > self.t
        head_bool = self.head('1') > self.t
        if head_bool and tail_bool: return len(self.model)
        else: return 0

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



