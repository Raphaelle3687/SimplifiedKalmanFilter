from scipy.stats._continuous_distns import _norm_pdf as pdf
from scipy.stats._continuous_distns import _norm_cdf as cdf
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
        val=(z*pdf(z)*cdf(z)+pdf(z)**2)/(cdf(z)**2)
        return val

    def infer(self, epsilon, beta, var0=0, iter=0, switchIndex=[]):

        switch=False
        if len(switchIndex)>0:
            switch=True
            when=switchIndex[0]
            switchIndex=switchIndex[1]

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

                self.probs[i, j] = self.getProbs(theta_I, xij,var_I, Y[i][j])

                var=np.dot(xij*var_I, xij)
                tempTheta=theta_I
                indicator=xij*xij
                V=self.functions.L1(theta_I, xij, Y[i][j])*self.scale
                theta_I=theta_I+xij*V*(var_I)/(np.sqrt(self.scale**2+var))
                var_I = var_I*(1 - indicator*self.W(tempTheta, xij) * (var_I) / (self.scale ** 2 + var))

            if i+1==len(X):
                break

            self.data.parametersMean[i+1]=theta_I
            self.data.parametersVar[i+1]=var_I

        return self.probs


    def PH(self, theta, x, V):
        var = np.dot(x * V, x)
        z=np.dot(theta, x)/(np.sqrt(self.scale**2+var))
        return cdf(z)

    def PA(self, theta, x, V):
        var=np.dot(x*V, x)
        z = np.dot(-theta, x) / (np.sqrt(self.scale ** 2 + var))
        return cdf(z)

    def getProbs(self, theta, x, V, d):
        PH = self.PH(theta, x, V)
        PA=self.PA(theta, x, V)
        return [PH, PA]