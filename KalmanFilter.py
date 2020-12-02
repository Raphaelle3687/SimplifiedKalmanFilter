import numpy as np
from scipy.stats import norm
import math
class KF:
    def __init__(self, data, model):
        self.data=data
        self.model=model
        self.data.setCov(True)
        if data.dim!=model.yDim:
            raise Exception("Dimension of the output of the model should match that of the data")

    def infer(self, alpha, beta, iter=1):

        X=self.data.input
        Y=self.data.output

        self.data.setVar0(self.model.scale)
        if self.data.cov==False:
            self.data.setCov(True)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            theta_I = beta * paramMean[i]
            varC_I = beta ** 2 * paramVar[i] + (np.identity(paramVar[i].shape[0]) * alpha)

            for j, xij in enumerate(Xi):

                for k in range(0, iter):

                    gt=self.model.L1(theta_I, xij, Y[i][j], add=0)
                    ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                    VtINV=np.linalg.inv(varC_I)
                    A=np.matmul(xij.reshape(len(xij), 1), xij.reshape(1, len(xij)))
                    Vt=np.linalg.inv ( ht*A + VtINV )
                    temp= xij*gt + xij*np.dot(ht*xij.transpose(), theta_I) + np.dot(VtINV, beta*paramMean[i])
                    theta_I= np.dot(Vt, temp)

                paramMean[i+1]=theta_I

                ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                Vt = np.linalg.inv(ht * A + VtINV)
                paramVar[i + 1] = Vt




