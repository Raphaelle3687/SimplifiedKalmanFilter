import numpy as np
from scipy.stats import norm
import math
class Elo:
    def __init__(self, data, sigma):
        self.data=data
        self.sigma=sigma
        if data.dim!=2:
            raise Exception("This data is incompatible for the Elo algorithm")

    def F(self,theta, x):
        z = np.dot(theta, x) / self.sigma
        return (1)/(1+10**(-z))

    def G(self,theta, x, y):
        sigmaPrim=self.sigma/(np.log(10))
        h=y[0]
        return (h-self.F(theta, x))/sigmaPrim

    def infer(self, K, misc, iter=1, var0=0):
        self.fit(K)

    def fit(self, K):

        X=self.data.input
        Y=self.data.output

        for i, Xi in enumerate(X):
            if i+1==len(X):
                break
            prevMean = self.data.parametersMean[i]
            self.data.parametersMean[i + 1]=prevMean
            for j, xij in enumerate(Xi):
                grad=self.G(xij, prevMean, Y[i][j])
                self.data.parametersMean[i+1]+=K*xij*grad

    def getProbs(self, theta, x, V, d):
        pW = self.F(theta, x)
        return [pW, 1-pW]

    def getMeanLS(self, start=None, end=None, P=None):
        LS2 = 0
        if start == None:
            start = 0
        if end == None:
            end = len(self.data.input)

        if P is None:
            P = self.data.output

        for i in range(start, end):

            Xi = self.data.input[i]

            LS1 = 0
            for j, xij in enumerate(Xi):

                    pW = self.F(self.data.parametersMean[i], xij)
                    W = P[i][j][0]
                    L=P[i][j][1]
                    LS1 -= (W*np.log(pW)+L*np.log(1-pW))

            LS2 += (LS1 / (j + 1))

        return LS2 / (end - start)