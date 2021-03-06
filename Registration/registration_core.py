# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:42:32 2019

@author: Dominic
"""

import numpy as np
from builtins import super
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors


def initialize_sigma2(X, Y):
    (N, D), (M, _)  = X.shape, Y.shape
    diff = X[np.newaxis,...] - Y[:,np.newaxis,:]
    err  = diff * diff
    return np.sum(err) / (D * M * N)

class expectation_maximization_registration(object):
    def __init__(self, X, Y, sigma2=None, max_iterations=1000, tolerance=0.001, w=0, *args, **kwargs):
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.X, self.Y = X, Y
        self.sigma2 = sigma2
        (self.N, self.D),(self.M, _) = self.X.shape, self.Y.shape
        self.tolerance      = tolerance
        self.w              = w
        self.max_iterations = max_iterations
        self.iteration      = 0
        self.err            = self.tolerance + 1
        self.P, self.Pt1, self.P1 = np.zeros((self.M, self.N)), np.zeros((self.N, )), np.zeros((self.M, ))
        self.Np             = 0

    def register(self, callback=lambda **kwargs: None):
        self. TY = self.transform_point_cloud(self.Y)
        if self.sigma2 is None:
            self.sigma2 = initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D/2 * np.log(self.sigma2)
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.do_iteration()
            if callable(callback):
                callback(iteration=self.iteration, error=self.err, X=self.X, Y=self.TY)
        return self.TY, self.registration_parameters()

    def registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def do_iteration(self):
        self.e_step()
        self.m_step()
        self.iteration += 1

    def e_step(self):
        """
        Perform E step of registration
        """
        #compute distance between points
        diff     = self.X[:,np.newaxis] - self.TY
        diff     = diff*diff
        
        #store dists
        P  = np.sum(diff, axis=-1).T
        
        #compute constant factor in denominator
        c = ((2 * np.pi * self.sigma2) ** (self.D / 2)) * (self.w / (1 - self.w)) * self.M / self.N
        
        #compute denominator
        P = np.exp(-P / (2 * self.sigma2)) 
        denom = np.sum(P, axis=0)
        denom[denom==0] = np.finfo(float).eps
        denom += c
        
        #compute P
        self.P = np.divide(P,denom)

    def m_step(self):
        """
        Perform M step of registration
        """
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1  = np.sum(self.P, axis=1)
        self.Np  = np.sum(self.P1)
        
        self.solve()
        self.TY = self.transform_point_cloud(self.Y)
        self.update_variance()




class simple_affine_registration(expectation_maximization_registration):
    def __init__(self, B=None, t=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = np.eye(self.D) if B is None else B
        self.t = np.zeros([1, self.D]) if t is True else t

    def solve(self):
        """
        Main bulk if the m step calculations specific to type of registration
        """
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.Xhat = self.X - muX
        Yhat = self.Y - muY
                
        self.A = np.transpose(self.Xhat) @ np.transpose(self.P) @ Yhat
        
        self.YPY = np.transpose(Yhat) @ np.diag(self.P1) @ Yhat
        
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        if self.t is not None:
            self.t = np.transpose(muX) - np.transpose(self.B) @ np.transpose(muY)
    def transform_point_cloud(self, Y):
        """
        Transform a given point cloud
        """
        if self.t is None:
            return Y @ self.B
        else:
            return Y @ self.B + self.t
    def inverse_transform_point_cloud(self,Y):
        """
        Inverse transform a given point cloud
        """
        return (Y - self.t) @ np.linalg.inv(self.B) 

    def update_variance(self):
        """
        Compute new sigma
        """
        qprev = self.q
        
        trAB     = np.trace(self.A @ self.B)
        xPx      = np.transpose(self.Pt1) @ np.sum(self.Xhat*self.Xhat, axis=1)
        self.q   = (xPx - 2 * trAB + np.trace(self.B @ self.YPY @ self.B)) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)
        
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def registration_parameters(self):
        return self.B, self.t
    
    
    
    
    
    
class notranslation_affine_registration(expectation_maximization_registration):
    def __init__(self, B=None, t=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = np.eye(self.D) if B is None else B
        #self.t = np.zeros([1, self.D]) if t is True else t

    def solve(self):
        """
        Main bulk if the m step calculations specific to type of registration
        """
        #muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        #muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.Xhat = self.X #- muX
        Yhat = self.Y #- muY
                
        self.A = np.transpose(self.Xhat) @ np.transpose(self.P) @ Yhat
        
        self.YPY = np.transpose(Yhat) @ np.diag(self.P1) @ Yhat
        
        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        #if self.t is not None:
        #    self.t = np.transpose(muX) - np.transpose(self.B) @ np.transpose(muY)
    def transform_point_cloud(self, Y):
        """
        Transform a given point cloud
        """
        #if self.t is None:
        return Y @ self.B
        #else:
        #return Y @ self.B + self.t
    def inverse_transform_point_cloud(self,Y):
        """
        Inverse transform a given point cloud
        """
        return (Y) @ np.linalg.inv(self.B) 

    def update_variance(self):
        """
        Compute new sigma
        """
        qprev = self.q
        
        trAB     = np.trace(self.A @ self.B)
        xPx      = np.transpose(self.Pt1) @ np.sum(self.Xhat*self.Xhat, axis=1)
        self.q   = (xPx - 2 * trAB + np.trace(self.B @ self.YPY @ self.B)) / (2 * self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2)
        self.err = np.abs(self.q - qprev)

        self.sigma2 = (xPx - trAB) / (self.Np * self.D)
        
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

    def registration_parameters(self):
        return self.B#, self.t
    
    
    
    
    
def gaussian_kernel(Y, beta):
    (M, D) = Y.shape
    XX = np.reshape(Y, (1, M, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, M, 1))
    diff = XX-YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta))

class deformable_registration(expectation_maximization_registration):
    def __init__(self, alpha=2, beta=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha         = 2 if alpha is None else alpha
        self.beta          = 2 if alpha is None else beta
        self.W             = np.zeros((self.M, self.D))
        self.G             = gaussian_kernel(self.Y, self.beta)

    def solve(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx      = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy      = np.dot(np.transpose(self.P1),  np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY    = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def registration_parameters(self):
        return self.G, self.W

# Scikit minimizer anisotropic scaling ICP

class anisotropic_scaling_ICP:

    def __init__(self, q, n):
        self.q = q
        self.n = n

        self.get_initial_transform()

        print("First transform: ", self.transform)

        self.plot(plot_transform=True)

        # Update point cloud using first transform
        self.q = self.transform @ self.q
        self.plot()

        self.iters = 0



    def get_initial_transform(self):
        Cq = np.cov(self.q)
        Cn = np.cov(self.n)

        eig_q = np.linalg.eig(Cq)[0]
        eig_n = np.linalg.eig(Cn)[0]

        lam = np.sqrt(eig_q)
        mu = np.sqrt(eig_n)

        self.nu = 1 / len(lam) * np.sum(mu / lam)

        self.delta = 0.1 * self.nu

        s_01 = self.nu
        s_02 = self.nu

        self.s_low_lim = 0.9 * self.nu
        self.s_high_lim = 1.1 * self.nu

        self.transform = np.diag([s_01, s_02])

    def iterate(self):
        self.x_guess = np.diagonal(self.transform)
        self.get_correspondences()
        self.res = minimize(self.ls, self.x_guess, method='BFGS')
        next_transform = np.diag(self.res.x)
        self.transform = self.transform @ next_transform
        self.q = next_transform @ self.q
        self.iters += 1

    def ls(self, s_in):
        S = np.diag([s_in[0], s_in[1]])
        output = np.sum(np.linalg.norm((S @ self.q) - self.nns) ** 2)
        return output

    def plot(self, plot_transform=False):
        plt.figure()
        plt.plot(self.n[0, :], self.n[1, :], 'o', label="Reference")
        plt.plot(self.q[0, :], self.q[1, :], 'o', label="transformed")

        if plot_transform:
            plt.plot((self.transform @ self.q)[0, :], (self.transform @ self.q)[1, :], 'o', label="initial transform",
                     alpha=0.6)
        plt.legend()
        plt.show()

    def get_correspondences(self):
        nn = NearestNeighbors(n_neighbors=1, algorithm='auto')
        nn.fit(self.n.T)

        distances, indices = nn.kneighbors(self.q.T)

        nns = np.array([self.n[:, index[0]] for index in indices]).T
        self.nns = nns

        # plt.hist(distances)
        return distances, indices
