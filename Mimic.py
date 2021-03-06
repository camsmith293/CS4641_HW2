import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import stats
from sklearn.metrics import mutual_info_score

np.set_printoptions(precision=4)


class Mimic(object):
    """
    Usage: from mimicry import Mimic
    :param domain: list of tuples containing the min and max value for each parameter to be optimized, for a bit
    string, this would be [(0, 1)]*bit_string_length
    :param fitness_function: callable that will take a single instance of your optimization parameters and return
    a scalar fitness score
    :param samples: Number of samples to generate from the distribution each iteration
    :param percentile: Percentile of the distribution to keep after each iteration, default is 0.90
    """

    def __init__(self, domain, fitness_function, samples=1000, percentile=0.70, maximize=False, discreteValues=False):

        self.domain = domain
        self.samples = samples
        self.discreteValues = discreteValues
        initial_samples = np.array(self._generate_initial_samples())
        self.sample_set = SampleSet(initial_samples, fitness_function, maximize=maximize)
        self.fitness_function = fitness_function
        self.percentile = percentile
        self.maximize = maximize


    def fit(self):
        """
        Run this to perform one iteration of the Mimic algorithm
        :return: A tuple containing the list containing the top percentile of data points
                 and the theta value
        """

        samples = self.sample_set.get_percentile(self.percentile)[0]
        self.distribution = Distribution(samples)
        self.sample_set = SampleSet(
            self.distribution.generate_samples(self.samples),
            self.fitness_function,
            maximize=self.maximize
        )
        return self.sample_set.get_percentile(self.percentile)

    def _generate_initial_samples(self):
        return [self._generate_initial_sample() for i in range(self.samples)]

    def _generate_initial_sample(self):
        if not self.discreteValues:
            return [random.uniform(self.domain[i][0], self.domain[i][1])
                    for i in range(len(self.domain))]
        else:
            return [random.choice(self.domain)
                    for i in range(len(self.domain))]


class SampleSet(object):
    def __init__(self, samples, fitness_function, maximize=False):
        self.samples = samples
        self.fitness_function = fitness_function
        self.maximize = maximize

    def calculate_fitness(self):
        sorted_samples = sorted(
            self.samples,
            key=self.fitness_function,
            reverse=self.maximize,
        )
        return np.array(sorted_samples)

    def get_percentile(self, percentile):
        fit_samples = self.calculate_fitness()
        index = int(len(fit_samples) * percentile)
        return fit_samples[:index], self.fitness_function(fit_samples[index])


class Distribution(object):
    def __init__(self, samples):
        self.samples = samples
        #self.complete_graph = self._generate_mutual_information_graph()
        #self.spanning_graph = self._generate_spanning_graph()
        #self._generate_bayes_net()

    def generate_samples(self, number_to_generate):
        samples = list()
        for i in range(number_to_generate):
            samples.append(random.choice(self.samples.tolist()))
        return samples


if __name__ == "__main__":
    samples = [
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 0, 0, 0],
    ]

    distribution = Distribution(samples)

    distribution._generate_bayes_net()

    for node_ind in distribution.bayes_net.nodes():
            print(distribution.bayes_net.node[node_ind])

    pos = nx.spring_layout(distribution.spanning_graph)

    edge_labels = dict(
        [((u, v,), d['weight'])
         for u, v, d in distribution.spanning_graph.edges(data=True)]
    )

    nx.draw_networkx(distribution.spanning_graph, pos)
    nx.draw_networkx_edge_labels(
        distribution.spanning_graph,
        pos,
        edge_labels=edge_labels,
    )

    plt.show()