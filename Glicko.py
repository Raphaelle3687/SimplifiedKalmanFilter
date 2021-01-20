from scipy.stats import norm
import numpy as np
from Model import Model
class Glicko:
    def __init__(self, data, sigma):
        self.scale=sigma
        self.data=data
        if data.dim!=2:
            raise Exception("This data is incompatible for this implementation of the Glicko algorithm")

    def F(self, z):
        return 1 / (1 + 10 ** (-z))

    def f(self, z, v):
        a = np.log(10) / self.scale
        r=self.r(v)
        val=a**2*r**2*(self.F(r*z/self.scale))*(1-self.F(r*z/self.scale))
        return val

    def r(self, v):
        a=np.log(10)/self.scale
        return 1/np.sqrt(1+(3*v*a**2)/np.pi**2)

    def infer(self, epsilon, beta, var0=0, iter=0):
        a=np.log(10)/self.scale
        #beta=1
        if var0==0:
            var0=epsilon

        self.beta=beta
        self.epsilon=epsilon
        X=self.data.input
        Y=self.data.output

        if self.data.cov==True:
            self.data.setCov(False)

        self.data.setVar0(var0)

        for i, Xi in enumerate(X):

            if i+1==len(X):
                break
            theta_I = beta * self.data.parametersMean[i]
            var_I = (((beta**2)*self.data.parametersVar[i]) + epsilon)

            for j, xij in enumerate(Xi):

                play1=np.where(xij==1)
                play2=np.where(xij==-1)
                z=np.dot(xij,theta_I)
                v1=var_I[play1]
                v2=var_I[play2]
                var_I[play1]=v1*(1)/(1+v1*self.f(z, v2))
                var_I[play2] = v2 * (1) / (1 + v2 * self.f(-z, v1))
                theta_I[play1]=theta_I[play1]+a*v1*self.r(v2)*(Y[i][j][0]-self.F(self.r(v2)*z/self.scale))
                theta_I[play2] = theta_I[play2] + a * v2 * self.r(v1) * (Y[i][j][1] - self.F(self.r(v1) * -z / self.scale))


            self.data.parametersMean[i+1]=theta_I
            self.data.parametersVar[i+1]=var_I


    def PH(self, theta, x, V):
        var = np.dot(x * V, x)
        z=np.dot(x, theta)
        return self.F(self.r(var)*z/self.scale)


    def PA(self, theta, x, V):
        return 1-self.PH(theta, x, V)

    def getProbs(self, theta, x, V, d):
        PH = self.PH(theta, x, V)
        PA=self.PA(theta, x, V)
        return [PH, PA]