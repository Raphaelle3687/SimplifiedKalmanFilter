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

    def infer(self, alpha, beta, iter=1, var0=0):
        self.beta=beta
        self.alpha=alpha
        X=self.data.input
        Y=self.data.output

        if self.data.cov==True:
            self.data.setCov(False)

        if var0==0:
            var0=alpha

        self.data.setVar0(var0)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            theta_I = beta * paramMean[i]
            var_I = (((beta**2)*paramVar[i]) + alpha)
            for j, xij in enumerate(Xi):

                u = (var_I) * xij
                d = u * u
                s = np.dot(u.transpose(), xij)

                for k in range(0, iter):

                    gt=self.model.L1(theta_I, xij, Y[i][j], add=0)
                    ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                    theta_I=theta_I+(1/(1+ht*s))*u*gt

                ht = self.model.L2(theta_I, xij, Y[i][j], add=0)
                var_I = var_I - (d * (ht) / (1 + ht * s))

            paramMean[i+1]=theta_I
            paramVar[i + 1] = var_I


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





