import numpy as np
import math
from scipy.stats import norm
class Model:
    def __init__(self, mod, scale, args=None):
        self.yDim=None
        self.yFunctions=[]
        self.yFunctionsOrd1=[]
        self.yFunctionsOrd2=[]
        self.scale=scale
        #self.Loss=None
        #self.LossOrd1=None
        #self.LossOrd2=None

        if mod=="Elo":
            el=Elo(scale)
            self.yDim=el.dim
            self.yFunctions.append(el.FH); self.yFunctions.append(el.FA)
        elif mod=="EloDavidson":
            el=EloDavidson(scale, args[0])
            self.yDim = el.dim
            self.yFunctions.append(el.FH);self.yFunctions.append(el.FA); self.yFunctions.append(el.FD)
        elif mod=="RaoKupper":
            r=RaoKupper(scale, args[0])
            self.yDim=r.dim
            self.yFunctions.append(r.FH); self.yFunctions.append(r.FA); self.yFunctions.append(r.FD)
            self.yFunctionsOrd1.append(r.derLogFh); self.yFunctionsOrd1.append(r.derLogFa); self.yFunctionsOrd1.append(r.derLogFd)
            self.yFunctionsOrd2.append(r.HessHome); self.yFunctionsOrd2.append(r.HessAway); self.yFunctionsOrd2.append(r.HessDraw);

        elif mod=="Thurstone":
            r = Thurstone(scale)
            self.yDim = r.dim
            self.yFunctions.append(r.FH); self.yFunctions.append(r.FA);
            self.yFunctionsOrd1.append(r.derLogFh); self.yFunctionsOrd1.append(r.derLogFa);
            self.yFunctionsOrd2.append(r.hessianH); self.yFunctionsOrd2.append(r.hessianA);


    def costFunction(self,theta, x ,Y, add=0):

        cost=0
        for i,y in enumerate(Y):
            cost+=y*math.log(self.yFunctions[i](theta, x, add=add))
        return cost

    def L1(self, theta, x ,Y, add=0):

        val=0
        for i,y in enumerate(Y):
            val+=y*self.yFunctionsOrd1[i](theta, x, add=add)
        return val

    def L2(self, theta, x ,Y, add=0):
        val = 0
        for i, y in enumerate(Y):
            val += y * self.yFunctionsOrd2[i](theta, x, add=add)
        return val

class Elo:

    def __init__(self, scale):

        self.scale=scale
        self.dim=2

    def FH(self, theta, x, add=0):

        s=self.scale+add
        z=np.dot(theta, x)
        return ( 1 )/( 1+10**(z/s) )

    def FA(self, theta, x, add=0):
        return self.FH(-theta, x, add=add)

class EloDavidson:

    def __init__(self, scale, kappa):
        self.scale=scale
        self.kappa=kappa
        self.dim=3

    def FH(self, theta, x, add=0):
        s = self.scale + add
        z = np.dot(theta, x)
        return (10 ** (0.5*z / s)) / (10 ** (-0.5*z / s) + 10 ** (0.5*z / s)+self.kappa)

    def FA(self, theta, x , add=0):
        s = self.scale + add
        z = np.dot(theta, x)
        return (10 ** (-0.5 * z / s)) / (10 ** (-0.5 * z / s) + 10 ** (0.5 * z / s) + self.kappa)

    def FD(self, theta, x, add=0):
        s = self.scale + add
        z = np.dot(theta, x)
        return (self.kappa) / (10 ** (-0.5 * z / s) + 10 ** (0.5 * z / s) + self.kappa)

def sigmoid( x):
    return 1/(1+ math.exp(-x))

class BradleyTerry:
    def __init__(self, scale):
        self.scale=scale

class RaoKupper:
    def __init__(self, scale, kappa):
        self.scale=scale
        self.kappa=kappa
        self.dim=3

    def FH(self, theta, x, add=0):
        z=np.dot(theta, x)/(self.scale+add)
        return sigmoid(z-self.kappa)

    def FA(self, theta, x, add=0):
        z = np.dot(-theta, x) / (self.scale + add)
        return sigmoid(z - self.kappa)

    def FD(self, theta, x, add=0):

        return 1-self.FH(theta, x, add=add)-self.FA(theta, x , add=add)

    def derLogFh(self, theta, x, add=0):
        fH = np.dot(theta, x) / self.scale - self.kappa
        val = math.exp(-fH) / (1 + math.exp(-fH))
        return val/self.scale

    def derLogFa(self, theta, x, add=0):
        fA = np.dot(theta, x) / self.scale + self.kappa
        val = math.exp(fA) / (1 + math.exp(fA))
        return -val/self.scale


    def derFh(self, theta, x, add=0):
        fH=np.dot(theta, x)/self.scale - self.kappa
        val = math.exp(-fH) / (1 + math.exp(-fH)) ** 2
        return val/self.scale

    def derFa(self, theta, x, add=0):
        fA=np.dot(theta, x)/self.scale + self.kappa
        val = math.exp(fA) / (1 + math.exp(fA)) ** 2
        return -val/self.scale

    def derLogFd(self, theta, x, add=0):
        val = (1 / self.FD(theta, x, add=add)) * (-self.derFa(theta, x, add=add) - self.derFh(theta, x, add=add))
        return val

    def HessHome(self, theta, x, add=0):
        fH=np.dot(theta, x)/self.scale - self.kappa
        val = math.exp(-fH) / (1 + math.exp(-fH))**2
        return val/(self.scale**2)

    def HessAway(self, theta, x, add=0):
        fA=np.dot(theta, x)/self.scale + self.kappa
        val = math.exp(fA) / (1 + math.exp(fA)) ** 2
        return val/(self.scale**2)

    def HessDraw(self, theta, x, add=0):

        val=self.HessHome( theta, x, add=add)+self.HessAway( theta, x, add=add)
        return val/(self.scale**2)

class Thurstone:
    def __init__(self, scale):
        self.scale = scale
        self.dim=2

    def FH(self, theta, x, add=0):
        z = np.dot(theta, x) / math.sqrt((self.scale**2 + add))
        return norm.cdf(z)

    def FA(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2 + add))
        return norm.cdf(z)

    def derLogFh(self, theta, x, add=0):
        z = np.dot(theta, x) / math.sqrt((self.scale**2 + add))
        return (norm.pdf(z)/norm.cdf(z))/(self.scale+add)

    def derLogFa(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2 + add))
        return -(norm.pdf(z)/norm.cdf(z))/(self.scale+add)

    def hessianH(self, theta, x, add=0):
        z = np.dot(theta, x) / math.sqrt((self.scale**2 + add))
        val=(z*norm.pdf(z)*norm.cdf(z)+norm.pdf(z)**2)/(norm.cdf(z)**2)
        return val/(self.scale+add)**2

    def hessianA(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2 + add))
        val=(z*norm.pdf(z)*norm.cdf(z)+norm.pdf(z)**2)/(norm.cdf(z)**2)
        return val/(self.scale+add)**2