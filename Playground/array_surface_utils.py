import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar

import logging

logging.basicConfig(format='%(asctime)s.%(msecs)03d  {%(module)s} [%(funcName)s] -- %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
log = logging.getLogger(__name__)





def hessian_normal(vector):
    # ND array with root sum of squares for each normal
    root_sum_squares = np.sqrt(np.sum(np.square(vector)))
    out = vector / root_sum_squares

    return out, root_sum_squares


class Fake_CP:

    def __init__(self, crosstalk_matrix_box, axes, tolerance):
        """
        :param crosstalk_matrix_box: Device.shapes_list object (crosstalk_matrix_box)
        :param axes: Row values of the matrix between which to look for CP behaviour.
        :param tolerance: distance from line within which return True for CP

        Each row in the matrix is a normal vector for a hyperplane that represents a (barrier) gate
        At the junction of hypersurfaces representing adjacent barrier gates, dots exist
        """
        log.info("Fake CP object initialised")

        self.matrix = crosstalk_matrix_box.matrix
        self.maxes = crosstalk_matrix_box.max

        print(self.matrix)

        # Axes: create dot between matrix rows axes[0] (int) and axis[1] (int, axes[0]+1) rows (i.e. surfaces)
        self.axes = axes

        # X_0: Common point of all surfaces = corner of hypercube
        self.X_0 = np.linalg.inv(self.matrix).dot(self.maxes)
        log.info("Corner of hypercube calculated: {}".format(self.X_0))
        self.rows = np.array([self.matrix[i, :] for i in range(self.matrix.shape[0])])
        self.normals = [self.normalised(self.matrix[i, :]) for i in range(self.matrix.shape[0])]
        self.tolerance = tolerance

    def distance_to_plane(self, point, i):
        # Find the distance from the given point to the hyperplane defined by matrix row i
        v = self.normals[i]
        w = self.X_0 - point
        # Only absolute distance to the plane matters hence abs()
        return abs(w[np.newaxis].dot(v.T))

    def check_cp(self, point):
        # TODO: hardcoded. Update so this uses self.axis

        # point = np.array(list(point) + list([0, 0, 0])).squeeze()

        # If distance to both planes < tolerance, point is near junction (to within tolerance)
        # Separated out like this for debugging

        close_to_plane1 = self.distance_to_plane(point, self.axes[0]) < self.tolerance
        close_to_plane2 = self.distance_to_plane(point, self.axes[1]) < self.tolerance

        res = [close_to_plane1[0][0], close_to_plane2[0][0]]
        return np.all(res)

    def normalised(self, a, axis=-1, order=2):
        # Returns normalised vector
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def distance_to_line(self, t, point):
        # Returns the Euclidean distance between a point and an interface
        val = np.linalg.norm(self.X_0 + (t * self.v) - point)
        return val

    def check_cp_3DONLY(self, point):
        # If distance to line is less than tolerance, fake observed CP behaviour
        opt = minimize_scalar(self.distance_to_line, args=(point))
        return opt.fun < self.tolerance


class Euclid_to_line:

    def __init__(self, crosstalk_matrix_box, axes):
        """
        :param crosstalk_matrix_box: Device.shapes_list object (crosstalk_matrix_box)
        :param axes: Row values of the matrix between which to look for CP behaviour.
        """
        self.matrix = crosstalk_matrix_box.matrix
        self.maxes = crosstalk_matrix_box.max

        self.X_0 = np.linalg.inv(self.matrix).dot(self.maxes)
        self.normals = [self.matrix[i, :] for i in range(self.matrix.shape[0])]
        self.v = self.normalised(np.cross(self.normals[axes[0]], self.normals[axes[1]]))[0]

    def normalised(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def dist(self, point, t):
        return np.linalg.norm(self.X_0 + (t * self.v) - point)

    def distance_to_line(self, t, point):
        # Returns the Euclidean distance between a point and a hyperline
        val = np.linalg.norm(self.X_0 + (t * self.v) - point)
        return val


def define_crosstalk_matrix(dims=(3, 3), mean=0, stdev=1.2):
    gaussian = scipy.stats.norm(mean, stdev)
    maximum = gaussian.pdf(mean)

    def f(i, j):
        difference = i - j + np.random.normal(0, 0.2)
        return (gaussian.pdf(difference) / maximum)

    matrix = np.fromfunction(lambda i, j: f(i, j), dims, dtype='float32')

    return matrix


def check_interface(vec1, vec2, x, max, delta):
    vec = vec2 - vec1
    vec = np.cross(vec1, vec2)
    diff = max[1] - max[0]
    # TODO: check if abs is the right way of doing this
    return vec.T.dot(x[:, np.newaxis]) - (diff) <= delta
