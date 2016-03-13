from random import choice, randint
from pybrain.structure.evolvables.evolvable import Evolvable

class FourPeaks(Evolvable):
    def __init__(self):
        self.length = len(self.model)
        self.t = int(self.length/10)
        self.model = "" * self.length
        self.randomize()

    def f(self):
        tail = self.tail('0')
        head = self.head('1')
        return max(tail, head) + self.R()

    def tail(self, b):
        return (len(self.model) - len(self.model.rstrip(b)))

    def head(self, b):
        return len(self.model.split('0', 1)[0])

    def R(self):
        tail_bool = self.tail('0') > self.t
        head_bool = self.head('1') > self.t
        if head_bool and tail_bool: return len(self.model)
        else: return 0

    def mutate(self, **args):
        index = choice(list(range(self.length)))
        if self.model[index] is '0':
            self.model[index] = '1'
        else:
            self.model[index] = '0'

    def randomize(self):
        model_list = [randint(0,1) for b in range(self.length)]
        self.mode = ''.join(str(e) for e in model_list)

    def domain(self):
        return [0,1] * self.length



