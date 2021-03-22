import numpy as np
import math
class KalmanFilter:
    def __init__(self, data, model, mode): #KF, vSKF, sSKF, fSKF
        self.data=data
        self.model=model
        self.data.setCov(True)
        self.mode=mode
        if self.mode!="KF" and self.mode!="vSKF" and self.mode!="sSKF" and self.mode!="fSKF":
            raise Exception("Invalid mode")
        if data.dim!=model.yDim:
            raise Exception("Dimension of the output of the model should match that of the data")

    def infer(self, alpha, beta, var0=0, iter=1, switchIndex=[]):
        switch=False
        if len(switchIndex)>0:
            switch=True
            when=switchIndex[0]
            switchIndex=switchIndex[1]

        X=self.data.input
        Y=self.data.output

        self.probs=np.zeros(Y.shape)

        self.epsilon=alpha
        self.beta=beta

        if var0==0:
            var0=alpha
            if self.mode=="fSKF":
                print("YOU SHOULD PROVIDE A VALID VARIANCE")

        if self.data.cov==False:
            self.data.setCov(True)
        self.data.setVar0(var0)

        paramMean=self.data.parametersMean
        paramVar=self.data.parametersVar

        for i, Xi in enumerate(X):

            thetPrev = beta * paramMean[i]
            thetaI=thetPrev

            if self.mode=="fSKF":
                Vt=paramVar[i]
            else:
                varC_I = beta ** 2 * paramVar[i] + (np.identity(paramVar[i].shape[0]) * alpha)
                Vt = varC_I

            if switch==True and i==when:

                thetaI[switchIndex]=0
                for ind in switchIndex:
                    Vt[:,ind]=0
                    Vt[ind,:]=0
                    Vt[ind][ind]=var0
                if self.mode == "sSKF":
                    Vt = np.identity(len(Vt)) * np.mean(np.diag(Vt))

            for j, xij in enumerate(Xi):

                self.probs[i, j]=self.model.getProbs(thetaI, xij, Vt, Y[i][j])

                var=np.linalg.multi_dot([xij.transpose(), Vt, xij])
                gt = self.model.L1(thetaI, xij, Y[i][j], add=0)
                ht = self.model.L2(thetaI, xij, Y[i][j], add=0)

                for k in range(0, iter):
                    #+ht*(np.dot(xij, (thetaI-thetPrev)))

                    coeff=(gt)/(1+ht*var)
                    vv=np.matmul(Vt, xij)
                    toAdd = thetaI +  vv*coeff

                    thetaI=toAdd

                if self.mode!="fSKF":
                    #ht = self.model.L2(thetaI, xij, Y[i][j], add=0)

                    A=(Vt@xij).reshape(len(xij), 1)
                    B=(xij.reshape(1, len(xij))@Vt)
                    matrix=A@B

                    Vt = Vt-(matrix)*(ht)/(1+ht*var)

            if i+1==len(X):
                continue

            paramMean[i+1]=thetaI

            if self.mode=="KF":
                paramVar[i + 1] = Vt
            elif self.mode=="vSKF":
                paramVar[i+1]=np.identity(len(Vt))*Vt
            elif self.mode=="sSKF":
                paramVar[i+1]=np.identity(len(Vt))*np.mean(np.diag(Vt))
            elif self.mode=="fSKF":
                paramVar[i+1]=paramVar[i]

        return self.probs



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




