import numpy as np
import numpy.random as rand
class DataSet:

    def __init__(self, input, output):

        self.input=input#np array of np arrays x_t
        self.output=output#np array of y
        self.P=None
        self.dim=output.shape[2]
        self.cov=False
        if len(input)!=len(output):
            raise Exception("Length of the input and output should match")

        dimParam=[self.input.shape[0], self.input.shape[2]]

        self.parametersMean=np.zeros(dimParam)
        self.parametersVar=np.zeros(dimParam)

        self.syntheticOutput = None
        self.syntheticParameters = None

    def setCov(self, isCov):

        self.cov=isCov
        dim=self.parametersMean.shape
        if isCov:
            self.parametersVar=np.zeros([dim[0], dim[1], dim[1]])
        else:
            self.parametersVar=np.zeros([dim[0], dim[1]])


    def resetParam(self):
        dimParam=[self.input.shape[0], self.input.shape[2]]

        self.parametersMean=np.zeros(dimParam)
        self.parametersVar=np.zeros(dimParam)


    def getTrio(self, t):
        return (self.input[t], self.output[t], self.parametersMean[t])


    def yieldTrio(self, synt=False):
        for i in range(self.getSize()):
            if synt:
                yield self.getSynTrio(i)
            else:
                yield self.getTrio(i)

    def getSynTrio(self, t):
        return (self.input[t], self.syntheticOutput[t], self.syntheticParameters[t])

    def getY(self,t ):
        return self.output[t]

    def getX(self,t):
        return self.input[t]

    def getSize(self):
        return len(self.input)

    def generateSynthetic(self, alpha,beta,  model):
        raise Exception("Should be adapted to the dimensions used in total synthetic")
        self.syntheticParameters=np.zeros(self.parametersMean.shape)
        self.syntheticOutput=np.zeros(self.output.shape)

        covMat=np.identity(self.parametersMean.shape[1])*alpha
        for i, x in enumerate(self.input):

            probs=[f(self.syntheticParameters[i], x) for f in model.yFunctions]
            self.syntheticOutput[i][self.biasedDice(probs)]=1
            if i+1<len(self.input):
                self.syntheticParameters[i+1]=rand.multivariate_normal(self.syntheticParameters[i], covMat)
        data=DataSet(self.input, self.syntheticOutput)
        data.parametersMean=self.syntheticParameters
        return data


    def biasedDice(self, bias):
        x=rand.rand(1)
        sum=0
        for i, val in enumerate(bias):
            if x<sum+val:
                return i
            else:
                sum+=val

    def setVar0(self, var):

        if self.cov==True:
            self.parametersVar[0] = np.identity(self.parametersVar.shape[1])*var
        else:
            self.parametersVar[0] = (np.zeros(len(self.parametersVar[0]))*+var)
