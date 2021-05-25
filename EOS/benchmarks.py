import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class TestFunction:
    def plot_surface(self, n_points=50, elev=45, azim=35, figsize=(8, 8)):

        fn = self.__call__

        X = np.linspace(start=self.LB[0], stop=self.UB[0], num=n_points)
        Y = np.linspace(start=self.LB[1], stop=self.UB[1], num=n_points)
        X, Y = np.meshgrid(X, Y)
        z = np.array([fn(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
        Z = z.reshape((n_points, n_points))
        #
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection="3d")
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cm.binary,
            linewidth=1,
            antialiased=False,
            rstride=2,
            cstride=2,
            alpha=0.8,
        )
        ax.plot_wireframe(
            X,
            Y,
            Z,
            color="gray",
            linewidth=0.4,
            alpha=0.8,
            rstride=2,
            cstride=2,
        )
        ax.view_init(elev, azim)
        # return fig

    def plot_contour(self, n_points=50, figsize=(8, 8), levels=20):

        fn = self.__call__

        X = np.linspace(start=self.LB[0], stop=self.UB[0], num=n_points)
        Y = np.linspace(start=self.LB[1], stop=self.UB[1], num=n_points)
        X, Y = np.meshgrid(X, Y)
        z = np.array([fn(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())])
        Z = z.reshape((n_points, n_points))
        #
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.contour(X, Y, Z, colors="gray", levels=levels)
        ax.grid()

        for px, py in self.x_opt:
            plt.plot(
                px,
                py,
                "o",
                color="red",
                fillstyle="none",
                markersize=11,
                markeredgewidth=2,
            )
            plt.plot(
                px,
                py,
                ".",
                color="red",
            )

    def plot_trayectory(self, stats, i_run=0):

        x = [point[0] for point in stats.x_best_per_generation[i_run]]
        y = [point[1] for point in stats.x_best_per_generation[i_run]]

        self.plot_contour()

        plt.gca().plot(
            x,
            y,
            "o-k",
            alpha=0.5,
        )

        plt.gca().plot(
            x[0],
            y[0],
            "o",
            c="black",
            fillstyle="none",
            markersize=11,
            markeredgewidth=2,
        )

        plt.gca().plot(
            x[-1],
            y[-1],
            "o",
            c="red",
            fillstyle="none",
            markersize=11,
            markeredgewidth=2,
        )

        #  plt.show()


class Ackely(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-30] * n_dim
        self.UB = [30] * n_dim
        self.fn_opt = 0
        self.x_opt = [[0 for _ in range(n_dim)]]

    def __call__(self, x):
        p1 = 20 + np.exp(1) - 20 * np.exp(-0.2 * np.mean(x ** 2))
        p2 = np.exp(np.mean(np.cos(2 * np.pi * x)))
        return p1 - p2


class Sphere(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-5.12] * n_dim
        self.UB = [5.12] * n_dim
        self.fn_opt = 0
        self.x_opt = [[0 for _ in range(n_dim)]]

    def __call__(self, x):
        return np.sum(x ** 2)


class Rosenbrock(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-2.048] * n_dim
        self.UB = [+2.048] * n_dim
        self.fn_opt = 0
        self.x_opt = [[1 for _ in range(n_dim)]]

    def __call__(self, x):
        return np.sum(
            [
                100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2
                for i in range(self.n_dim - 1)
            ]
        )


class Step(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-30] * n_dim
        self.UB = [+30] * n_dim
        self.fn_opt = 0
        self.x_opt = [[0 for _ in range(n_dim)]]

    def __call__(self, x):
        return np.sum(np.floor(x + 0.5) ** 2)


class Griewank(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-100] * n_dim
        self.UB = [+100] * n_dim
        self.fn_opt = 0
        self.x_opt = [[0 for _ in range(n_dim)]]

    def __call__(self, x):
        a = np.sum(x ** 2) / 4000
        b = np.prod(np.cos(x / np.sqrt(np.arange(1, self.n_dim + 1))))
        return 1.0 + a + b


class Rastrigin(TestFunction):
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.LB = [-5.12] * n_dim
        self.UB = [+5.12] * n_dim
        self.fn_opt = 0
        self.x_opt = [[0 for _ in range(n_dim)]]

    def __call__(self, x):
        p1 = np.sum(x ** 2)
        p2 = 10 * np.sum(np.cos(2 * np.pi * x))
        return 10 * self.n_dim + p1 - p2
