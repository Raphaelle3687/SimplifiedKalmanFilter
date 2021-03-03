import numpy as np
from scipy.stats import norm
import math
class KalmanFilter:
    def __init__(self, data, model, mode): #KF, v-SKF, s-SKF, f-SKF
        self.data=data
        self.model=model
        self.data.setCov(True)
        self.mode=mode
        if data.dim!=model.yDim:
            raise Exception("Dimension of the output of the model should match that of the data")

    def infer(self, alpha, beta, iter=1, var0=0):

        X=self.data.input
        Y=self.data.output
        self.alpha=alpha
        self.beta=beta

        if var0==0:
            var0=alpha
            if self.mode=="f-SKF":
                print("YOU SHOULD PROVIDE A VALID VARIANCE")

        if self.data.cov==False:
            self.data.setCov(True)
        self.data.setVar0(var0)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            if i+1==len(X):
                continue

            thetPrev = beta * paramMean[i]
            thetaI=thetPrev
            if self.mode=="f-SKF":
                Vt=paramVar[i]
            else:
                varC_I = beta ** 2 * paramVar[i] + (np.identity(paramVar[i].shape[0]) * alpha)
                Vt = varC_I

            for j, xij in enumerate(Xi):

                var=np.linalg.multi_dot([xij.transpose(), Vt, xij])
                gt = self.model.L1(thetaI, xij, Y[i][j], add=0)
                ht = self.model.L2(thetaI, xij, Y[i][j], add=0)

                for k in range(0, iter):
                    #+ht*(np.dot(xij, (thetaI-thetPrev)))

                    coeff=(gt)/(1+ht*var)
                    toAdd = thetaI + np.matmul(Vt, xij) * coeff

                    thetaI=toAdd

                if self.mode!="f-SKF":
                    ht = self.model.L2(thetaI, xij, Y[i][j], add=0)

                    A=(Vt@xij).reshape(len(xij), 1)
                    B=(xij.reshape(1, len(xij))@Vt)
                    matrix=A@B

                    Vt = Vt-(matrix)*(ht)/(1+ht*var)

            paramMean[i+1]=thetaI

            if self.mode=="KF":
                paramVar[i + 1] = Vt
            elif self.mode=="v-SKF":
                paramVar[i+1]=np.identity(len(Vt))*Vt
            elif self.mode=="s-SKF":
                paramVar[i+1]=np.identity(len(Vt))*np.mean(np.diag(Vt))
            elif self.mode=="f-SKF":
                paramVar[i+1]=paramVar[i]



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




