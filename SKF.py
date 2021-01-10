import numpy as np
from scipy.stats import norm
import math
class SKF:
    def __init__(self, data, model):
        self.data=data
        self.model=model
        self.data.setCov(False)
        if data.dim!=model.yDim:
            raise Exception("Dimension of the output of the model should match that of the data")

    def infer(self, alpha, beta, iter=1):
        self.beta=beta
        self.alpha=alpha
        X=self.data.input
        Y=self.data.output

        if self.data.cov==True:
            self.data.setCov(False)

        self.data.setVar0(alpha)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            theta_I = beta * paramMean[i]
            var_I = (((beta**2)*paramVar[i]) + alpha)
            varI=var_I
            for j, xij in enumerate(Xi):

                u = (beta ** 2 * var_I + alpha) * xij #PROBLEM
                d = u * u
                s = np.dot(u.transpose(), xij)

                for k in range(0, iter):

                    gt=self.model.L1(theta_I, xij, Y[i][j], add=0)
                    ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                    theta_I=theta_I+(1/(1+ht*s))*u*gt

                paramMean[i+1]=theta_I

                ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                varI = varI - (d * (ht) / (1 + ht * s))
                paramVar[i + 1] = varI


    def microInfer(self, theta, V, x, y):

        u = (self.beta ** 2 * V + self.alpha) * x
        d = u * u
        s = np.dot(u.transpose(), x)

        gt = self.model.L1(theta, x, y)
        ht = self.model.L2(theta, x, y)
        newTheta = theta + (1 / (1 + ht * s)) * u * gt

        ht = self.model.L2(theta, x, y)
        newVar = V - (d * (ht) / (1 + ht * s))

        return newTheta, newVar





