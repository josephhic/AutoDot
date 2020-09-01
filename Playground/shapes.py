import numpy as np
import scipy.stats


def L2_norm(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


class Circle(object):
    def __init__(self, ndim, r=1000, origin=0.0):
        if np.isscalar(origin):
            self.origin = origin * np.ones(ndim)
        else:
            self.origin = np.array(origin)
        self.r = r
        self.ndim = ndim

    def __call__(self, x):
        return L2_norm(np.array(x) - self.origin) <= self.r


class Box(object):
    def __init__(self, ndim, a=-1000, b=0):
        if np.isscalar(a):
            a = a * np.ones(ndim)
        if np.isscalar(b):
            b = b * np.ones(ndim)

        a = np.array(a)
        b = np.array(b)

        if len(a) != ndim or len(b) != ndim:
            raise ValueError('Wrong dimensions for defining a box')
        if all(a < b):
            self.lb, self.ub = a, b
        elif all(a > b):
            self.lb, self.ub = b, a
        else:
            raise ValueError('Wrong points for defining a box')
        self.ndim = ndim

    def __call__(self, x):
        x = np.array(x)
        inside = np.logical_and(np.all(x > self.lb[np.newaxis, :], axis=-1),
                                np.all(x < self.ub[np.newaxis, :], axis=-1))
        return inside


class Leakage(object):
    def __init__(self, ndim, th_leak=-500, idx=0):
        self.th_leak = th_leak
        self.leak_gate = idx

    def __call__(self, x):
        x = np.array(x)
        leak = x[:, self.leak_gate] > self.th_leak
        return leak


# change
class Convexhull(object):
    def __init__(self, ndim, points=[-1000, 0, [-1000, 0], [0, -1000]]):
        # points: 2D array (num_points x ndim)
        for i, point in enumerate(points):
            if np.isscalar(point):
                points[i] = point * np.ones(ndim)

        points = np.array(points)

        from scipy.spatial import Delaunay
        self.hull = Delaunay(points)
        self.ndim = points.shape[1]

    def __call__(self, x):
        return self.hull.find_simplex(x) >= 0


class Crosstalk_box(Convexhull):
    def __init__(self, ndim, a=-1500, b=1500, a_prime=-1000):

        if np.isscalar(a):
            a = a * np.ones(ndim)
        if np.isscalar(b):
            b = b * np.ones(ndim)
        if np.isscalar(a_prime):
            a_prime = a_prime * np.ones(ndim)

        a = np.array(a)
        b = np.array(b)

        if np.any(a > b): raise ValueError('a should be less than b')

        vertices = np.array(np.meshgrid(*list(zip(a, b)))).T.reshape(-1, ndim)
        # Replace the first vertex with b
        vertices[0] = a_prime

        from scipy.spatial import Delaunay
        self.hull = Delaunay(vertices)
        self.ndim = vertices.shape[1]


class Crosstalk_matrix_box:

    def __init__(self, ndim, mean=0, stdev=1.2, max=[-1200, -1200, -1200]):
        """
        Creates a device hypersurface for a device with ndim (barrier) gates
        Crosstalk between gates is defined by crosstalk matrix (define_crosstalk_matrix)
        This matrix is randomly generated using gaussian from mean, stdev provided in init (from config).
        """

        self.mean = mean
        self.stdev = stdev
        self.ndim = ndim

        self.max = np.array(max)

        self.matrix = self.define_crosstalk_matrix(self.ndim, self.mean, self.stdev)



        """
        print("Matrix not randomly generated. Change in Playground/shapes.py")
        self.matrix = np.array([[1.00000000e+00, 2.25765077e-01, 1.25099259e-03, 8.05792338e-08,
                                 9.45403448e-13],
                                [2.66750478e-01, 1.00000000e+00, 3.26841059e-01, 4.12765083e-04,
                                 5.35202839e-08],
                                [2.03396254e-03, 1.85905049e-01, 1.00000000e+00, 1.43947865e-01,
                                 7.78828027e-04],
                                [8.45614836e-08, 2.73164211e-03, 2.09218832e-01, 1.00000000e+00,
                                 2.73003173e-01],
                                [4.61575699e-13, 1.89962878e-08, 3.65315897e-04, 3.68532330e-01,
                                 1.00000000e+00]])
        """

    def __call__(self, x):
        x = np.array(x)
        check = self.matrix.dot(x.T) > self.max[:, np.newaxis]
        shape = np.all(check[:, :], axis=0)
        return shape

    def define_crosstalk_matrix(self, ndim, mean, stdev):
        dims = (ndim, ndim)

        gaussian = scipy.stats.norm(mean, stdev)
        maximum = gaussian.pdf(mean)

        def f(i, j):
            # Create array of random values (0, 0.3) to add to crosstalk matrix
            randoms = np.random.random(dims) * 0.3

            # Add to index difference (i - j) to maintain gaussian
            # np.sign so that they shift away from mean not just towards higher value

            difference = i - j
            return gaussian.pdf(difference + (randoms * np.sign(j - i))) / maximum

        matrix = np.fromfunction(lambda i, j: f(i, j), dims, dtype='float32')

        # Plot the matrix that represents the crosstalk. Blues cmap to make it similar to the Volk qubyte paper
        import matplotlib.pyplot as plt
        plt.matshow(matrix, cmap='Blues')

        return matrix
