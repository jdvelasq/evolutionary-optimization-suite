from operator import itemgetter
import random
import numpy as np
from .individual import Individual


class Operator:
    def cloning(self, population):
        return [Individual(individual.copy()) for individual in population]


# ============================================================================
#
#  Map
#
# ============================================================================


class Mapper(Operator):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, population):
        population = self.cloning(population)
        for individual in population:
            individual = self.fn(individual)
        return population


# ============================================================================
#
#  Cloning
#
# ============================================================================


class Cloning:
    def __call__(self, population, idx=None):
        clones = [Individual(individual.copy()) for individual in population]
        if idx is not None:
            clones = [population[i] for i in idx]
        return clones


# ============================================================================
#
#  Selection
#
# ============================================================================


class SelectionBest(Operator):
    #
    # Selecciona los k mejores individuos de la población
    #
    def __init__(self, k=None, as_index=False):
        self.k = k
        self.as_index = as_index

    def __call__(self, population):

        population = self.cloning(population)

        if self.k is None:
            k = len(population)
        else:
            k = self.k

        idx = [(idx, individual.fn_x) for idx, individual in enumerate(population)]
        idx = sorted(idx, key=itemgetter(1), reverse=False)
        idx = [i for i, _ in idx]
        idx = idx[:k]
        if self.as_index is False:
            return [population[i] for i in idx]
        return idx


class SelectionWorst(Operator):
    #
    # Selecciona los k peores individuos de la población
    #
    def __init__(self, k=None, as_index=False):
        self.k = k
        self.as_index = as_index

    def __call__(self, population):

        population = self.cloning(population)

        if self.k is None:
            k = len(population)
        else:
            k = self.k

        idx = [(idx, individual.fn_x) for idx, individual in enumerate(population)]
        idx = sorted(idx, key=itemgetter(1), reverse=False)
        idx = [i for i, _ in idx]
        idx = idx[-k:]
        if self.as_index is False:
            return [population[i] for i in idx]
        return idx


class SelectionRandom(Operator):
    #
    # Selecciona aleatoriamente k individuos de la población
    #
    def __init__(self, k, seed=None, as_index=False):

        self.k = k
        self.as_index = as_index

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        population = self.cloning(population)

        idx = self.rng.choice(
            np.arange(len(population)),
            size=self.k,
            replace=True,
        )
        if self.as_index is False:
            return [population[i] for i in idx]
        return idx


class SelectionTournament(Operator):
    def __init__(self, k, tournsize, seed=None):

        self.k = k
        self.tournsize = tournsize

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        population = self.cloning(population)

        popsize = len(population)
        fitness = []

        for idx, individual in enumerate(population):

            oponents = list(range(popsize))
            oponents.remove(idx)
            oponents = self.rng.choice(
                oponents,
                size=self.k,
                replace=False,
            )

            fn_x_values = [population[oponent].fn_x for oponent in oponents]
            fn_x = individual.fn_x
            winings = sum([0 if fn_x_value < fn_x else 1 for fn_x_value in fn_x_values])
            fitness.append((idx, winings))

        fitness = sorted(fitness, key=itemgetter(1), reverse=True)
        idx = [i for i, _ in fitness]
        idx = idx[: self.k]
        return [population[i] for i in idx]


class SelectionRoulette(Operator):
    def __init__(self, k, seed=None):
        self.k = k

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        population = self.cloning(population)

        #
        # Ordena la población de peor a mejor
        #
        select = SelectionBest()
        population = select(population)
        population.reverse()

        N = len(population)
        roulette = np.arange(1, N + 1)
        roulette = roulette / np.sum(roulette)
        roulette = np.cumsum(roulette)

        selected = []
        for _ in range(self.k):

            rnd = self.rng.uniform()
            for idx, spin in enumerate(roulette):
                if rnd <= spin:
                    selected.append(idx)
                    break

        return [population[idx] for idx in selected]


# ============================================================================
#
#  Crossover
#
# ============================================================================


class CrossoverUniform(Operator):
    def __init__(self, probability=0.5, seed=None, both=False):

        self.probability = probability
        self.both = both

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, parent1, parent2):

        parent1 = self.cloning(parent1)
        parent2 = self.cloning(parent2)

        n_dim = len(parent1[0].x)

        #
        # Cruza las x entre padres
        #
        rnd_uniform = [self.rng.uniform(size=n_dim) for _ in range(len(parent1))]
        offspring1 = [
            Individual(
                {
                    "x": np.where(rnd < self.probability, p1.x, p2.x),
                    "fn_x": None,
                }
            )
            for rnd, p1, p2 in zip(rnd_uniform, parent1, parent2)
        ]
        offspring2 = [
            Individual(
                {
                    "x": np.where(rnd < self.probability, p2.x, p1.x),
                    "fn_x": None,
                }
            )
            for rnd, p1, p2 in zip(rnd_uniform, parent1, parent2)
        ]

        #
        # Cruza los sigmas entre padres si existen
        #
        if "ES_sigma" in parent1[0].keys():

            rand = [self.rng.uniform(size=n_dim) for _ in range(len(parent1))]

            for r, o1, o2, p1, p2 in zip(
                rand, offspring1, offspring2, parent1, parent2
            ):
                o1["ES_sigma"] = np.where(
                    r < self.probability, p1.ES_sigma, p2.ES_sigma
                )
                o2["ES_sigma"] = np.where(
                    r < self.probability, p2.ES_sigma, p1.ES_sigma
                )

        if self.both is True:
            return offspring1 + offspring2

        return [
            o1 if self.rng.uniform() < 0.5 else o2
            for o1, o2 in zip(offspring1, offspring2)
        ]


class CrossoverOnePoint(Operator):
    def __init__(self, both=True, seed=None):
        self.both = both
        self.n_dim = None

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, parent1, parent2):

        if self.n_dim is None:
            self.n_dim = len(parent1[0].x)

        offspring1 = self.cloning(parent1)
        offspring2 = self.cloning(parent2)

        for o1, o2 in zip(offspring1, offspring2):

            point = self.rng.integers(low=1, high=self.n_dim - 1, size=1)[0]

            x1 = o1.x[:point]
            x2 = o2.x[:point]

            o1.x[:point] = x2
            o2.x[:point] = x1

            o1.fn_x = None
            o2.fn_x = None

        if self.both is True:
            return offspring1 + offspring2

        return offspring1


# ============================================================================
#
#  Mutation
#
# ============================================================================


class MutationGaussian(Operator):
    def __init__(
        self,
        probability=1.0,
        sigma=0.1,
        seed=None,
    ):
        self.probability = probability
        self.sigma = sigma

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        n_dim = len(population[0].x)

        population = self.cloning(population)

        for individual in population:

            individual.x = individual.x + self.sigma * self.rng.normal(size=n_dim)
            individual.fn_x = None

        return population


class MutationES(Operator):
    def __init__(
        self,
        probability=1.0,
        seed=None,
    ):
        self.probability = probability

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        #
        # Número de dimensiones
        #
        n_dim = len(population[0].x)

        population = self.cloning(population)

        for individual in population:
            #
            # Se mutan las variables x del problema
            #
            if self.rng.uniform() < self.probability:
                individual.x = individual.x + individual.ES_sigma * self.rng.normal(
                    size=n_dim
                )

        return population


class MutationSelfAdaptiveES(Operator):
    def __init__(
        self,
        probability=1.0,
        sigma_min=0.01,
        sigma_max=10,
        seed=None,
    ):
        self.probability = probability
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        #
        # Número de dimensiones
        #
        n_dim = len(population[0].x)

        #
        # Constante definida por el algoritmo
        #
        tau = 1.0 / np.power(2.0, 1.0 / n_dim)

        for individual in population:
            #
            # Muta los parámetros de estrategia
            #
            individual.ES_sigma = individual.ES_sigma * np.exp(
                tau * self.rng.normal(size=n_dim)
            )
            individual.ES_sigma = np.where(
                individual.ES_sigma < self.sigma_max,
                individual.ES_sigma,
                self.sigma_max,
            )
            individual.ES_sigma = np.where(
                individual.ES_sigma > self.sigma_min,
                individual.ES_sigma,
                self.sigma_min,
            )
            #
            # Muta las variables independientes
            #
            individual.x = individual.x + individual.ES_sigma * self.rng.normal(
                size=n_dim
            )

        return population


class MutationEP(Operator):
    def __init__(
        self,
        beta=1.0,
        gamma=0.0,
        seed=None,
    ):
        self.beta = beta
        self.gamma = gamma

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        population = self.cloning(population)

        #
        # Número de dimensiones
        #
        n_dim = len(population[0].x)

        #
        # El individuo tiene un vector de sigmas, una
        # por cada dimensión del vector x.
        #
        for individual in population:
            #
            # computa el sigma como una función de fn_x
            #
            sigma = np.sqrt(self.beta * individual.fn_x + self.gamma)

            #
            # Luego muta las x con el sigma computado
            #
            individual.x = individual.x + sigma * self.rng.normal(size=n_dim)
            individual.fn_x = None

        return population


class MutationMetaEP(Operator):
    def __init__(
        self,
        c=1.0,
        sigma_min=0.000001,
        sigma_max=4,
        probability=1.0,
        seed=None,
    ):
        self.c = c
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.probability = float(probability)

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def __call__(self, population):

        population = self.cloning(population)

        #
        # Número de dimensiones
        #
        n_dim = len(population[0].x)

        #
        # El individuo tiene un vector de sigmas, una
        # por cada dimensión del vector x.
        #
        for individual in population:
            #
            # Muta los parámetros de estrategia
            #
            individual.EP_sigma = individual.EP_sigma + self.rng.normal(
                size=n_dim
            ) * np.sqrt(self.c * individual.EP_sigma)

            #
            # Verifica que el sigma este entre los límites admisibles
            #
            individual.EP_sigma = np.where(
                individual.EP_sigma <= self.sigma_max,
                individual.EP_sigma,
                self.sigma_max,
            )
            individual.EP_sigma = np.where(
                individual.EP_sigma > self.sigma_min,
                individual.EP_sigma,
                self.sigma_min,
            )
            #
            # Luego muta las x con los nuevos sigmas
            #
            individual.x = individual.x + individual.EP_sigma * self.rng.normal(
                size=n_dim
            )

            individual.fn_x = None

        return population


class MutationFlipBit:
    def __init__(self, probability, seed=None):
        self.probability = probability
        self.n_dim = None

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def cloning(self, population):
        return [Individual(individual.copy()) for individual in population]

    def __call__(self, population):

        population = self.cloning(population)

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        for individual in population:
            individual.x = np.where(
                self.rng.uniform() <= self.probability, 1 - individual.x, individual.x
            )

        return population