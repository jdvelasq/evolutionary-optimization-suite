import numpy as np
import matplotlib.pyplot as plt


class Monitor:
    def __init__(self):
        #
        # Estadísticos a almacenar por generación
        #
        self.fn_x_avg = {}
        self.fn_x_std = {}
        self.fn_x_min = {}
        self.fn_x_max = {}
        self.fn_x_best = {}
        self.fn_x_worst = {}

        #
        # Mejores resultados por corrida por generación
        #
        self.x_best_per_generation = {}

        #
        # Mejores resultados encontrados
        #
        self.global_x_opt = None
        self.global_fn_x_opt = None

        #
        # Contador para la corrida actual
        #
        self.i_run = -1

    def new_run(self):
        #
        self.i_run += 1
        #
        self.fn_x_avg[self.i_run] = []
        self.fn_x_std[self.i_run] = []
        self.fn_x_min[self.i_run] = []
        self.fn_x_max[self.i_run] = []
        self.fn_x_best[self.i_run] = []
        self.fn_x_worst[self.i_run] = []
        #
        self.x_best_per_generation[self.i_run] = []

    def __call__(self, population):

        if self.i_run == -1:
            self.new_run()

        #
        # Evalua la función objetivo
        #
        fn_x = [individual.fn_x for individual in population]

        #
        # Cómputa los estadísticos para la población actual
        #
        self.fn_x_avg[self.i_run].append(np.mean(fn_x))
        self.fn_x_std[self.i_run].append(np.std(fn_x))
        self.fn_x_min[self.i_run].append(np.min(fn_x))
        self.fn_x_max[self.i_run].append(np.max(fn_x))
        self.fn_x_best[self.i_run].append(np.min(self.fn_x_min[self.i_run]))
        self.fn_x_worst[self.i_run].append(np.max(self.fn_x_max[self.i_run]))

        #
        # Mejor punto encontrado en la generación actual
        #
        argmin = np.argmin(fn_x)
        self.x_best_per_generation[self.i_run].append(population[argmin].x.copy())

        #
        # Mejor punto encontrado en todas las corridas
        #
        if self.global_fn_x_opt is None or min(fn_x) < self.global_fn_x_opt:
            self.global_fn_x_opt = min(fn_x)
            self.global_x_opt = population[argmin].x

    def plot_stats(self, i_run=0, figsize=(10, 6)):

        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        n = list(range(len(self.fn_x_avg[self.i_run])))
        plt.plot(n, self.fn_x_worst[self.i_run], "--g", label="Worst", alpha=0.6)
        plt.plot(n, self.fn_x_best[self.i_run], "--r", label="Best", alpha=0.6)
        plt.plot(n, self.fn_x_min[self.i_run], "-k", label="Min", alpha=1.0)
        plt.plot(n, self.fn_x_max[self.i_run], "-k", label="Max", alpha=1.0)
        plt.plot(n, self.fn_x_avg[self.i_run], "-", c="blue", label="Avg", alpha=1.0)
        plt.yscale("log")
        plt.legend()


def plot_performance(stats, legends=None, figsize=(10, 6)):

    fig = plt.figure(figsize=figsize)

    strs = ["-k", "--r", "--.b", "-b", "--k", "--.r"]

    if legends is None:
        legends = ["Experiment {}".format(i) for i in len(stats)]

    for i_obj, obj in enumerate(stats):

        fn_x_min_sum = [0] * len(obj.fn_x_min[0])
        for i_run in obj.fn_x_min.keys():
            y = obj.fn_x_min[i_run]
            fn_x_min_sum = [values + accum for values, accum in zip(y, fn_x_min_sum)]
        n = len(obj.fn_x_min.keys())
        fn_x_min_avg = [value / float(n) for value in fn_x_min_sum]

        plt.plot(
            list(range(len(fn_x_min_sum))),
            fn_x_min_avg,
            strs[i_obj],
            label=legends[i_obj],
        )

    plt.legend()
