from scipy.stats import norm
import numpy as np
from Model import Model
class Glicko:
    def __init__(self, data, sigma):
        self.scale=sigma
        self.data=data
        if data.dim!=2:
            raise Exception("This data is incompatible for this implementation of the Glicko algorithm")

        self.functions=Model("BradleyTerry", sigma)

    def E(self, z):
        return 1 / (1 + 10 ** (-z))

    def f(self, z, v):
        q = np.log(10) / self.scale
        g=self.g(v)
        val=q**2*g**2*(self.E(g*z/self.scale))*(1-self.E(g*z/self.scale))
        return val

    def g(self, v):
        q=np.log(10)/self.scale
        return 1/np.sqrt(1+(3*v*q**2)/np.pi**2)

    def infer(self, epsilon, beta, var0=0, iter=0, switchIndex=[]):

        switch=False
        if len(switchIndex)>0:
            switch=True
            when=switchIndex[0]
            switchIndex=switchIndex[1]

        q=np.log(10)/self.scale
        #beta=1
        if var0==0:
            var0=epsilon

        self.beta=beta
        self.epsilon=epsilon
        X=self.data.input
        Y=self.data.output

        self.probs = np.zeros(Y.shape)

        if self.data.cov==True:
            self.data.setCov(False)

        self.data.setVar0(var0)

        for i, Xi in enumerate(X):

            theta_I = beta * self.data.parametersMean[i]
            var_I = (((beta**2)*self.data.parametersVar[i]) + epsilon)

            if switch==True and i==when:

                theta_I[switchIndex]=0
                var_I[switchIndex]=var0

            for j, xij in enumerate(Xi):

                self.probs[i, j] = self.getProbs(theta_I, xij, var_I, Y[i][j])

                play1=np.where(xij==1)
                play2=np.where(xij==-1)
                z=np.dot(xij,theta_I)
                v1=var_I[play1]
                v2=var_I[play2]

                sig1Rev=(1/v1 + self.f(z, v2))
                sig2Rev=(1/v2 + self.f(z, v1))

                var_I[play1] = 1/sig1Rev
                var_I[play2] = 1/sig2Rev

                coeff1=self.g(v2)
                coeff2=(Y[i][j][0]-self.E(self.g(v2)*z/self.scale) )

                theta_I[play1]=theta_I[play1] + q * coeff1 * (Y[i][j][0]-self.E(self.g(v2)*z/self.scale) )/sig1Rev
                theta_I[play2] = theta_I[play2] + q * self.g(v1) * (Y[i][j][1] - self.E(self.g(v1) * -z / self.scale))/sig2Rev

            if i+1==len(X):
                break

            self.data.parametersMean[i+1]=theta_I
            self.data.parametersVar[i+1]=var_I

        return self.probs


    def PH(self, theta, x, V):
        var = np.dot(x * V, x)
        z=np.dot(x, theta)
        return self.F(self.r(var)*z/self.scale)


    def PA(self, theta, x, V):
        return 1-self.PH(theta, x, V)

    def getProbs(self, theta, x, V, d):

        return self.functions.getProbs(theta, x, V, d)

        PH = self.PH(theta, x, V)
        PA=self.PA(theta, x, V)
        return [PH, PA]