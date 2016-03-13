from random import choice, randint
from pybrain.structure.evolvables.evolvable import Evolvable

class kColors(Evolvable):

    def __init__(self):
        self.colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black']

        self.states = ['Andhra', 'Karnataka', 'TamilNadu', 'Kerala']

        self.neighbors = {}
        self.neighbors['Andhra'] = ['Karnataka', 'TamilNadu']
        self.neighbors['Karnataka'] = ['Andhra', 'TamilNadu', 'Kerala']
        self.neighbors['TamilNadu'] = ['Andhra', 'Karnataka', 'Kerala']
        self.neighbors['Kerala'] = ['Karnataka', 'TamilNadu']

        self.model = {}
        self.randomize()

    def promising(self, state, color):
        for neighbor in self.neighbors.get(state):
            color_of_neighbor = self.model.get(neighbor)
            if color_of_neighbor == color:
                return False

        return True

    def get_color_for_state(self, state):
        for color in self.colors:
            if self.promising(state, color):
                return color

    def f(self, assignments):
        conflicts = 0
        for i in range(assignments):
            if not self.promising(self.states[i], assignments[i]): conflicts += 1
        return conflicts

    def mutate(self, **args):
        state = choice(self.states)
        exclusionary = self.colors - self.model[state]
        self.model[state] = choice(exclusionary)

    def randomize(self):
        for state in self.states:
            self.model[state] = choice(self.colors)

    def domain(self):
        return self.colors * len(self.states)

