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

        X=self.data.input
        Y=self.data.output

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            theta_I = beta * paramMean[i]
            var_I = (((beta**2)*paramVar[i]) + alpha)

            for j, xij in enumerate(Xi):

                u = (beta ** 2 * var_I + alpha) * xij
                d = u * u
                s = np.dot(u.transpose(), xij)

                for k in range(0, iter):

                    gt=self.model.L1(theta_I, xij, Y[i][j], add=0)
                    ht=self.model.L2(theta_I, xij, Y[i][j], add=0) #minus? I gueeess not clear
                    theta_I=theta_I+(1/(1+ht*s))*u*gt

                paramMean[i+1]=theta_I

                ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                varI = var_I - (d * (ht) / (1 + ht * s))
                paramVar[i + 1] = varI




