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

    def infer(self, alpha, beta, iter=1, var0=0):

        X=self.data.input
        Y=self.data.output
        self.alpha=alpha
        self.beta=beta

        if var0==0:
            var0=alpha

        if self.data.cov==False:
            self.data.setCov(True)
        self.data.setVar0(var0)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            theta_I = beta * paramMean[i]
            varC_I = beta ** 2 * paramVar[i] + (np.identity(paramVar[i].shape[0]) * alpha)
            Vt=varC_I
            for j, xij in enumerate(Xi):

                for k in range(0, iter):

                    gt=self.model.L1(theta_I, xij, Y[i][j], add=0)
                    ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                    A=np.matmul(xij.reshape(len(xij), 1), xij.reshape(1, len(xij)))
                    VtINV = np.linalg.inv(Vt)
                    temp= xij*gt + xij*np.dot(ht*xij.transpose(), theta_I) + np.dot(VtINV, theta_I)
                    toAdd=np.matmul(Vt, temp)
                    theta_I=toAdd
                Vt = np.linalg.inv(ht * A + VtINV)

            paramMean[i+1]=theta_I
            paramVar[i + 1] = Vt
                #ht=self.model.L2(theta_I, xij, Y[i][j], add=0)
                #Vt = np.linalg.inv(ht * A + VtINV)

    def microInfer(self, theta, V, x, y):
        theta = self.beta * theta
        V = self.beta ** 2 * V + (np.identity(V.shape[0]) * self.alpha)

        gt = self.model.L1(theta, x, y)
        ht = self.model.L2(theta, x,y)
        VtINV = np.linalg.inv(V)
        A = np.matmul(x.reshape(len(x), 1), x.reshape(1, len(x)))
        Vt = np.linalg.inv(ht * A + VtINV)
        temp = x * gt + x * np.dot(ht * x.transpose(), theta) + np.dot(VtINV, theta)
        theta = np.matmul(Vt, temp)
        return theta, Vt




