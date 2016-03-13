from random import choice, randint
from pybrain.structure.evolvables.evolvable import Evolvable

colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black']
states = ['Andhra', 'Karnataka', 'TamilNadu', 'Kerala']

def fitness_kcolors(evaluable):
        conflicts = 0
        for i in range(len(states)):
            state = states[i]
            conflicts += evaluable.promising(state, evaluable.model[state])
        return conflicts

def fitness_kcolors_GA(evaluable):
        temp = kColors()
        for i in range(len(evaluable)):
            temp.model[states[i]] = evaluable[i]
        return fitness_kcolors(temp)

class kColors(Evolvable):

    def __init__(self):
        self.colors = colors

        self.states = states

        self.neighbors = {}
        self.neighbors['Andhra'] = ['Karnataka', 'TamilNadu']
        self.neighbors['Karnataka'] = ['Andhra', 'TamilNadu', 'Kerala']
        self.neighbors['TamilNadu'] = ['Andhra', 'Karnataka', 'Kerala']
        self.neighbors['Kerala'] = ['Karnataka', 'TamilNadu']

        self.model = {}
        self.randomize()

    def promising(self, state, color):
        collisions = 0
        for neighbor in self.neighbors.get(state):
            color_of_neighbor = self.model.get(neighbor)
            if color_of_neighbor == color: collisions += 1

        return collisions

    def get_color_for_state(self, state):
        for color in self.colors:
            if self.promising(state, color):
                return color

    def mutate(self, **args):
        state = choice(self.states)
        exclusionary = [elem for elem in self.colors if elem != self.model[state]]
        self.model[state] = choice(exclusionary)

    def randomize(self):
        for state in self.states:
            self.model[state] = choice(self.colors)

    def domain(self):
        return self.colors * len(self.states)

