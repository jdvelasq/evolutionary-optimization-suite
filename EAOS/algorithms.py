import numpy as np
from .individual import Individual
from .operators import SelectionBest


class Algorithm:
    def __init__(self, fn, seed=None):

        self.fn = fn
        self.n_dim = None

        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    def random_direction(self):
        w = self.rng.uniform(low=-1, high=+1, size=self.n_dim)
        m = np.linalg.norm(w)
        return w / m

    def cloning(self, population):
        return [Individual(individual.copy()) for individual in population]


class GradientDescendent(Algorithm):
    def __init__(self, fn, learning_rate, delta_x=0.001):
        super().__init__(fn=fn)
        self.delta_x = delta_x
        self.learning_rate = learning_rate

    def __call__(self, population):
        def gradient(individual):
            x = individual.x
            n_dim = len(x)
            gd = np.zeros(n_dim)
            fn_x = self.fn(x)
            for i_dim in range(n_dim):
                x_plus_delta = x.copy()
                x_plus_delta[i_dim] += self.delta_x
                fn_x_plus_delta = self.fn(x_plus_delta)
                gd[i_dim] = (fn_x_plus_delta - fn_x) / self.delta_x
            return gd

        population = self.cloning(population)

        for individual in population:
            gd = gradient(individual)
            individual.x = individual.x - self.learning_rate * gd
            individual.fn_x = self.fn(individual.x)

        return population


class CoordinateSearch(Algorithm):
    #
    # Propueto por Fermi y Metropolis en 1952. Fue diseñado
    # para solucionar sistemas complejos de ecuaciones en
    # fisica nuclear
    #
    def __init__(self, fn, delta=0.5):
        super().__init__(fn=fn)
        self.delta = delta

    def __call__(self, population):

        population = self.cloning(population)

        #
        # Asume que ya fue evaluada la función objetivo
        #  para toda la población. Busca a lo largo de cada
        #  eje hasta encontrar un punto mejor
        #
        for individual in population:

            if "CS_delta" not in individual.keys():
                individual["CS_delta"] = self.delta

            x = individual.x
            fn_x = individual.fn_x

            for i_coordinate in range(len(x)):

                x_left = x.copy()
                x_left[i_coordinate] = x_left[i_coordinate] - individual.CS_delta
                fn_x_left = self.fn(x_left)

                x_right = x.copy()
                x_right[i_coordinate] = x_right[i_coordinate] + individual.CS_delta
                fn_x_right = self.fn(x_right)

                if fn_x_left > fn_x and fn_x < fn_x_right:

                    #
                    # Encontró un punto de minima entre
                    #  x_left y x_right
                    #
                    for n in range(20):

                        l = x_left + 0.382 * (x_right - x_left)
                        u = x_left + 0.618 * (x_right - x_left)

                        fl = self.fn(l)
                        fu = self.fn(u)

                        if fl > fu:
                            x_left = l
                            fn_x_left = fl
                        else:
                            x_right = u
                            fn_x_right = fu

                    x_mean = 0.5 * (x_left + x_right)
                    fn_x_mean = self.fn(x_mean)

                    if fn_x_left < fn_x:
                        x = x_left
                        fn_x = fn_x_left

                    if fn_x_right < fn_x:
                        x = x_right
                        fn_x = fn_x_right

                    if fn_x_mean < fn_x:
                        x = x_mean
                        fn_x = fn_x_mean

                else:

                    #
                    # El punto de mínima está en uno de los extremos
                    #
                    if fn_x_left < fn_x:
                        x = x_left
                        fn_x = fn_x_left
                    if fn_x_right < fn_x:
                        x = x_right
                        fn_x = fn_x_right

            individual.x = x
            individual.fn_x = fn_x

        return population


class LocalSearch(Algorithm):
    #
    # Desarrollada por Hooke & Jeeves en 1962. Combina la
    # búsqueda lineal con la explotación de las direcciones
    # encontradas.
    #
    def __init__(self, fn, delta0=0.5, reduction_factor=0.9):
        super().__init__(fn=fn)
        self.delta0 = delta0
        self.reduction_factor = reduction_factor

    def __call__(self, population):

        population = self.cloning(population)

        #
        # Asume que ya fue evaluada la función objetivo
        #  para toda la población. Busca a lo largo de cada
        #  eje hasta encontrar un punto mejor y luego toma
        #  tomar una dirección diagonal para avanzar.
        #

        for individual in population:

            if "LS_delta" not in individual.keys():
                individual["LS_delta"] = self.delta0

            x_base = individual.x.copy()
            fn_x_base = individual.fn_x
            x = x_base.copy()
            fn_x = fn_x_base

            for i_coordinate in range(len(x)):

                x_left = x.copy()
                x_left[i_coordinate] = x_left[i_coordinate] - individual.LS_delta
                fn_x_left = self.fn(x_left)

                if fn_x_left < fn_x:
                    x = x_left
                    fn_x = fn_x_left
                else:
                    x_right = x.copy()
                    x_right[i_coordinate] = x_right[i_coordinate] + individual.LS_delta
                    fn_x_right = self.fn(x_right)

                    if fn_x_right < fn_x:
                        x = x_right
                        fn_x = fn_x_right

            #
            # En este punto  x y fn(x) son los puntos más
            #  bajos encontrados usando un ciclo de búsqueda
            # por coordenadas. Intenta avanzar en la
            # dirección de descenso encontrada.
            #
            x_next = 2 * x - x_base
            fn_x_next = self.fn(x_next)

            #
            # Si el nuevo punto es más bajo actualiza x y
            #  fn(x)
            #
            if fn_x_next < fn_x:
                x = x_next
                fn_x = fn_x_next

            #
            # Verifica si el ciclo permitio encontrar un
            #  punto más bajo. Si no lo encuentra, reduce el
            # valor del delta de las coordenadas cíclicas
            #
            if fn_x < fn_x_base:
                individual.x = x
                individual.fn_x = fn_x
            else:
                individual.LS_delta *= self.reduction_factor

        return population


class BacterialChemotaxis(Algorithm):
    #
    # Propuesto por Muller, Airgahi, Marchetto y Koumoustsakos
    # en el año 2000. Es un algoritmo de búsqueda local
    # elitista. Se base en la imitación de como las bacterias
    #  son atraidas a los mejores ambientes (a nivel macro)
    # y con el comportamiento de las colonias a nivel micro.
    #
    def __init__(self, fn, rho=0.5, seed=None):
        super().__init__(fn=fn, seed=seed)
        #
        # tamaño del avance
        #
        self.rho = rho

    def __call__(self, population):

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        population = self.cloning(population)

        for individual in population:

            if "BC_rho" not in individual.keys():
                individual["BC_rho"] = self.rho

            x_base = individual.x.copy()
            fn_x_base = individual.fn_x

            #
            # Genera un vector con dirección aleatoria
            # y radio rho
            #
            v = individual.BC_rho * self.random_direction()

            while True:

                #
                # Avanza en la dirección aleatoria. Si
                #  fn(x) mejora se continua avanzando
                # en esa misma dirección
                #
                x = x_base + v
                fn_x = self.fn(x)

                if fn_x < fn_x_base:
                    x_base = x
                    fn_x_base = fn_x
                else:
                    break

            individual.x = x_base.copy()
            individual.fn_x = fn_x_base

        return population


class SimulatedAnnealing(Algorithm):
    def __init__(self, fn, t_init=0.5, t_min=0.001, M=1000, seed=None):
        super().__init__(fn=fn, seed=seed)
        self.t_init = t_init
        self.t_min = t_min
        self.M = M
        self.factor = None

    def __call__(self, population):

        population = self.cloning(population)

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        if self.factor is None:
            self.factor = np.power(self.t_min / self.t_init, 1.0 / self.M)

        for individual in population:

            if "SA_t_current" not in individual.keys():
                individual["SA_t_current"] = self.t_init

            x = individual.x.copy()
            fn_x = individual.fn_x

            v = self.random_direction()
            x_next = x + self.rng.uniform() * individual.SA_t_current * v
            fn_x_next = self.fn(x_next)

            if self.rng.uniform() < 1 / (
                1
                + np.exp(
                    max(-50, min(50, (fn_x_next - fn_x) / individual.SA_t_current))
                )
            ):
                individual.x = x_next
                individual.fn_x = fn_x_next

            individual.SA_t_current = max(
                self.t_min, self.factor * individual.SA_t_current
            )

        return population


class ThresholdAcceptance(Algorithm):
    def __init__(self, fn, threshold=10, R=1.0, M=1000, seed=None):
        super().__init__(fn=fn, seed=seed)
        self.threshold = threshold
        self.R = R
        self.M = M
        self.factor = None

    def __call__(self, population):

        population = self.cloning(population)

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        if self.factor is None:
            if self.threshold > 0:
                self.factor = np.power(1e-6 / self.threshold, 1.0 / self.M)
            else:
                self.factor = 1

        for individual in population:

            if "TA_threshold" not in individual.keys():
                individual["TA_threshold"] = self.threshold

            x = individual.x.copy()
            fn_x = individual.fn_x

            v = self.random_direction()
            x_next = x + self.rng.uniform() * self.R * v
            fn_x_next = self.fn(x_next)

            if (fn_x_next - fn_x) <= individual.TA_threshold:
                individual.x = x_next
                individual.fn_x = fn_x_next
                individual.TA_threshold = individual.TA_threshold * self.factor

        return population


class DifferentialEvolution(Algorithm):
    def __init__(self, fn, LB, UB, crossover_rate=0.9, stepsize=0.8, seed=None):
        super().__init__(fn=fn, seed=seed)
        self.LB = LB
        self.UB = UB
        self.crossover_rate = crossover_rate
        self.stepsize = stepsize
        self.n_dim = None

    def __call__(self, population):

        population = self.cloning(population)

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        for i_individual, individual in enumerate(population):

            agents = np.arange(len(population))
            agents = np.delete(agents, i_individual)
            agents = self.rng.choice(agents, size=3)

            agent_a = population[agents[0]].x
            agent_b = population[agents[1]].x
            agent_c = population[agents[2]].x

            random_index = self.rng.integers(self.n_dim, size=1)[0]

            x = individual.x

            mutant = agent_a + self.stepsize * (agent_b - agent_c)

            for i_dim in range(self.n_dim):

                if self.rng.uniform() < self.crossover_rate or i_dim == random_index:

                    x[i_dim] = mutant[i_dim]
                    x[i_dim] = max(self.LB[i_dim], min(self.UB[i_dim], x[i_dim]))

            fn_x = self.fn(x)

            if fn_x < individual.fn_x:
                individual.x = x
                individual.fn_x = fn_x

        return population


class ParticleSwarmOptimization(Algorithm):
    def __init__(
        self,
        fn,
        velocity_max=0.5,
        cognition_learning_rate=2.05,
        social_learning_rate=2.05,
        seed=None,
    ):
        super().__init__(fn=fn, seed=seed)
        self.n_dim = None
        self.velocity_max = velocity_max
        self.cognition_learning_rate = cognition_learning_rate
        self.social_learning_rate = social_learning_rate

    def __call__(self, population):

        if self.n_dim is None:
            self.n_dim = len(population[0].x)

        population = self.cloning(population)

        #
        # Si es la primera iteración, crea los parámetros
        # del individuo
        #
        if "PSO_best_local_x" not in population[0].keys():
            #
            # Inicializa las variables requiridas por el algoritmo
            #
            for individual in population:
                individual.PSO_best_local_x = individual.x
                individual.PSO_best_local_fn_x = individual.fn_x
                individual.PSO_velocity = self.rng.uniform(
                    low=-self.velocity_max, high=self.velocity_max, size=self.n_dim
                )

        #
        # Obtiene el mejor individuo de la generación actual
        #
        select = SelectionBest(k=1)
        best = select(population)[0]

        #
        # Modifica la posición de cada individuo de la población
        #
        for individual in population:

            phi1 = self.rng.uniform(
                low=0, high=self.cognition_learning_rate, size=self.n_dim
            )
            phi2 = self.rng.uniform(
                low=0, high=self.social_learning_rate, size=self.n_dim
            )

            velocity = (
                individual.PSO_velocity
                + phi1 * (individual.PSO_best_local_x - individual.x)
                + phi2 * (best.x - individual.x)
            )
            m_velocity = np.linalg.norm(velocity)

            if m_velocity > self.velocity_max:
                velocity = velocity / m_velocity * self.velocity_max

            velocity = np.where(
                np.abs(velocity) >= self.velocity_max,
                np.sign(velocity) * self.velocity_max,
                velocity,
            )

            individual.x += velocity
            individual.fn_x = self.fn(individual.x)
            individual.PSO_velocity = velocity

            if individual.fn_x < individual.PSO_best_local_fn_x:
                individual.PSO_best_local_fn_x = individual.fn_x
                individual.PSO_best_local_x = individual.x

        return population
