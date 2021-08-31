import numpy as np
import math
#from scipy.stats import norm
from scipy.integrate import quad
from scipy.special import roots_hermite
from scipy.stats._continuous_distns import _norm_pdf as pdf
from scipy.stats._continuous_distns import _norm_cdf as cdf
class Model:
    def __init__(self, mod, scale, args=None):
        self.yDim=None
        self.yFunctions=[]
        self.yFunctionsOrd1=[]
        self.yFunctionsOrd2=[]
        self.estimates=[]
        self.scale=scale
        self.mode=""
        self.model=mod
        #self.Loss=None
        #self.LossOrd1=None
        #self.LossOrd2=None

        if mod=="Elo": #NOT FINISHED, equivalent to BradleyTerry
            el=Elo(scale)
            self.yDim=el.dim
            self.yFunctions.append(el.FH); self.yFunctions.append(el.FA)

        elif mod=="EloDavidson":#NOT FINISHED
            el=EloDavidson(scale, args[0])
            self.yDim = el.dim
            self.yFunctions.append(el.FH);self.yFunctions.append(el.FA); self.yFunctions.append(el.FD)

        elif mod=="RaoKupper":#NOT FINISHED
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
            self.estimates.append(r.estimatePH); self.estimates.append(r.estimatePA) #we have an exact function for the probability in this case
            self.mode="discrete"

        elif mod=="BradleyTerry":
            r=BradleyTerry(scale)
            self.yDim=r.dim
            self.yFunctions.append(r.FH); self.yFunctions.append(r.FA);
            self.yFunctionsOrd1.append(r.derLogFh); self.yFunctionsOrd1.append(r.derLogFa);
            self.yFunctionsOrd2.append(r.hessianH); self.yFunctionsOrd2.append(r.hessianA);
            self.mode = "discrete"

        elif mod=="Gaussian":
            r=Gaussian(scale)
            self.yDim=r.dim
            self.yFunctions=r.P
            self.yFunctionsOrd1=r.derLogP
            self.yFunctionsOrd2=r.hessianP
            self.estimates=r.estimateP #we have an exact function and don't want to estimate the probabilities
            self.mode="continuous"



    def costFunction(self,theta, x ,Y, add=0):

        cost=0
        if self.mode=="discrete":
            for i,y in enumerate(Y):
                cost+=y*math.log(self.yFunctions[i](theta, x, add=add))
        elif self.mode=="continuous":
            cost+=math.log(self.yFunctions(theta, x, Y))
        return cost

    def L1(self, theta, x ,Y, add=0):

        val=0
        if self.mode=="discrete":
            for i,y in enumerate(Y):
                val+=y*self.yFunctionsOrd1[i](theta, x, add=add)
        elif self.mode=="continuous":
            val+=self.yFunctionsOrd1(theta, x, Y)
        return val

    def L2(self, theta, x ,Y, add=0):
        val = 0
        if self.mode=="discrete":
            for i, y in enumerate(Y):
                val += y * self.yFunctionsOrd2[i](theta, x, add=add)
        elif self.mode=="continuous":
            val+=self.yFunctionsOrd2(theta, x, Y)
        return val

    #y here simply means for which outcome the probability is estimated
    def discreteEstimate2(self, theta, x, V, y):

        L=self.yFunctions[y](theta, x)
        g=self.yFunctionsOrd1[y](theta, x)
        h=self.yFunctionsOrd2[y](theta, x)

        if V.ndim==1:
            V=np.diag(V)
        var=np.linalg.multi_dot([x.transpose(), V, x])

        prob= np.exp( (g**2 *var)/(2*(h*var+1)) )* L/(np.sqrt(var*h+1))
        return prob

    def discreteEstimate(self, theta, x, V, y):
        L=self.yFunctions[y]
        var = np.linalg.multi_dot([x.transpose(), V, x])
        n=20
        [root, weights]=roots_hermite(n)
        estimate=0;
        for i, r in enumerate(root):
            thet=np.append(theta, np.sqrt(2*var)*r)
            sched=np.append(x, 1)
            estimate+=weights[i]*L(thet, sched)
        estimate*=1/np.sqrt(np.pi)
        return estimate


    #here d is the observation, in the continuous case the probability depends on it
    def continuousEstimate(self, theta, x, V, d):
        pass

    def getProbs(self, theta,x, V, d, isy=True):

        probs=[]

        if self.mode=="discrete":

            if isy==True:
                for i in range(self.yDim):
                    probs.append(self.yFunctions[i](theta, x))
                return probs


            if len(self.estimates)!=0:

                for k, f in enumerate(self.estimates):
                    probs.append(f(theta, x, V))

            else:
                for i in range(self.yDim):
                    probs.append(self.discreteEstimate(theta, x, V, i))

        elif self.mode=="continuous":

            if len(self.estimates)!=0:
                probs.append(self.estimates(theta, x, V, d))

            else:
                probs.append(self.continuousEstimate(theta, x, V, d))

        if self.mode=="discrete":
            probs=np.array(probs)
            probs=probs/np.sum(probs)
        return probs

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
        self.dim=2

    def FH(self, theta, x, add=0):
        s=self.scale+add
        z=np.dot(theta, x)/s
        return 1/(1+10**(-z))

    def FA(self, theta, x, add=0):
        return self.FH(-theta, x, add=add)

    def derLogFh(self, theta, x, add=0):
        val=(np.log(10)/self.scale)*(1-self.FH(theta, x))
        return val

    def derLogFa(self, theta, x, add=0):
        val = (np.log(10) / self.scale) * (0 - self.FH(theta, x))
        return val

    def hessianH(self, theta, x, add=0):
        z=np.dot(theta, x)/self.scale
        val=(np.log(10)/self.scale)**2 *(1/(10**(z/2)+10**(-z/2))**2)
        return val

    def hessianA(self, theta, x, add=0):
        return self.hessianH(theta, x)

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
        return cdf(z)

    def FA(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2 + add))
        return cdf(z)

    def estimatePH(self, theta, x, V):
        if V.ndim==1:
            V=np.diag(V)
        var=np.linalg.multi_dot([x.transpose(), V, x])
        return self.FH(theta, x, add=var)

    def estimatePA(self, theta, x, V):
        if V.ndim==1: #handles the SKF case where the Cov matrix is a vector of the diagonal elements
            V=np.diag(V)

        var=np.linalg.multi_dot([x.transpose(), V, x])
        return self.FA(theta, x,add=var)

    def derLogFh(self, theta, x, add=0):
        z = np.dot(theta, x) / math.sqrt((self.scale**2+add))
        return (pdf(z)/cdf(z))/(self.scale)

    def derLogFa(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2+add))
        return -(pdf(z)/cdf(z))/(self.scale)

    def hessianH(self, theta, x, add=0):
        z = np.dot(theta, x) / math.sqrt((self.scale**2+add))
        val=(z*pdf(z)*cdf(z)+pdf(z)**2)/(cdf(z)**2)
        return val/(self.scale)**2

    def hessianA(self, theta, x, add=0):
        z = np.dot(-theta, x) / math.sqrt((self.scale**2+add))
        val=(z*pdf(z)*cdf(z)+pdf(z)**2)/(cdf(z)**2)
        return val/(self.scale)**2

class Gaussian:
    def __init__(self, scale):
        self.scale = scale
        self.dim=1

    def P(self, theta, x, y):
        z=np.dot(theta, x)
        return pdf((z-y)/self.scale)

    def estimateP(self, theta, x, V, y):
        if V.ndim==1:
            V=np.diag(V)
        var=np.linalg.multi_dot([x.transpose(), V, x])
        realVar=var+self.scale**2
        mean=np.dot(theta, x)
        #z=(y-mean)/np.sqrt(realVar)
        return pdf(y, loc=mean, scale=np.sqrt(realVar))


    def derLogP(self, theta, x, y):
        z=np.dot(theta, x)
        val=(y-z)/self.scale**2
        return val

    def hessianP(self, theta, x, y):
        return 1/self.scale**2