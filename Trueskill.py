from scipy.stats import norm
import numpy as np
from Model import Model
class Trueskill:
    def __init__(self, data, sigma):
        self.scale=sigma
        self.data=data
        if data.dim!=2:
            raise Exception("This data is incompatible for this implementation of the Trueskill algorithm")

        self.functions=Model("Thurstone", self.scale)

    def W(self, theta, x):
        z = np.dot(theta, x) / self.scale
        val=(z*norm.pdf(z)*norm.cdf(z)+norm.pdf(z)**2)/(norm.cdf(z)**2)
        return val

    def infer(self, epsilon, beta, var0=0, iter=0):
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

                var=np.dot(xij*var_I, xij)
                tempTheta=theta_I
                indicator=xij*xij
                V=self.functions.L1(theta_I, xij, Y[i][j])*self.scale
                theta_I=theta_I+xij*V*(var_I)/(np.sqrt(self.scale**2+var))
                var_I = var_I*(1 - indicator*self.W(tempTheta, xij) * (var_I) / (self.scale ** 2 + var))

            self.data.parametersMean[i+1]=theta_I
            self.data.parametersVar[i+1]=var_I


    def PH(self, theta, x, V):
        var = np.dot(x * V, x)
        z=np.dot(theta, x)/(np.sqrt(self.scale**2+var))
        return norm.cdf(z)

    def PA(self, theta, x, V):
        var=np.dot(x*V, x)
        z = np.dot(-theta, x) / (np.sqrt(self.scale ** 2 + var))
        return norm.cdf(z)

    def getProbs(self, theta, x, V, d):
        PH = self.PH(theta, x, V)
        PA=self.PA(theta, x, V)
        return [PH, PA]