import math
import numpy.random as rand
import numpy as np
from Model import Model
import random
def getLSOnInfer(infer, P=None):

    paramM=infer.data.parametersMean
    paramV=infer.data.parametersVar
    #paramV=None
    input=infer.data.input
    output=infer.data.output

    return getMeanLS(input, output, paramM, infer.model, paramVar=paramV, P=P)


def getMeanLS(input, output, paramMean, model,paramVar=None, start=None, end=None , P=None):
    LS2 = 0
    if start == None:
        start = 0
    if end == None:
        end = len(input)

    if P is None:
        P = output

    for i in range(start, end):

        Xi =input[i]

        LS1 = 0
        for j, xij in enumerate(Xi):

            if paramVar is None:
                s=0
            else:
                s = np.dot(xij.transpose(), xij * paramVar[i])

            for k, f in enumerate(model.yFunctions):
                p=f(paramMean[i], xij, add=s)
                print(p)
                LS1 -= P[i][j][k] * math.log(p)


        LS2 += (LS1/(j+1))

    return LS2 / (end - start)


def genCompleteSynth(M, D, alpha, beta, model=Model("Thurstone", 2)):

    indexA=np.arange(0, M).tolist()

    skills=np.zeros([D, M])
    X=np.zeros([D,int(M/2), M])
    Y=np.zeros([D,int(M/2), model.yDim])
    P=np.zeros([D,int(M/2), model.yDim])

    covMat = np.identity(skills.shape[1])*((alpha)/(1-beta**2))
    skills[0]=rand.multivariate_normal(skills[0], covMat)
    covMat=np.identity(skills.shape[1])*alpha

    for i in range(0, D):

        if i!=0:
            skills[i]=rand.multivariate_normal(beta*skills[i-1], covMat)

        newSchedule=random.sample(indexA, len(indexA))

        for j in range(0, int(M/2)):

            xij=np.zeros(M)
            xij[newSchedule.pop()]=1
            xij[newSchedule.pop()]=-1
            pij=np.zeros(model.yDim)

            for k, f in enumerate(model.yFunctions):
                pij[k]=  f (beta*skills[i], xij)

            yij=np.zeros(model.yDim)
            yij[biasedDice(pij)]=1
            X[i][j]=xij
            P[i][j]=pij
            Y[i][j]=yij

    return X,Y,P,skills


def biasedDice(bias):
    x=rand.rand(1)
    sum=0
    for i, val in enumerate(bias):
        if x<sum+val:
            return i
        else:
            sum+=val


def optiBetaAlpa(trainingSets):



    for sets in trainingSets:
        pass

def goldenSearch(function, functionArgs, initIntervals, reduction=0.05):

    phi=(1+math.sqrt(5))/2

    valuesDict={}
    start=initIntervals[0][0]
    end=initIntervals[0][1]
    cInit=end-start
    c=cInit

    Intervals = []
    for i in range(0,len(functionArgs)):
        x1 = initIntervals[i][0]
        x2 = initIntervals[i][1]
        x3 = (x2-x1) * (phi - 1) / phi + x1
        Intervals.append([x1, x3, x2])

    while c > reduction*cInit:

        combinations=[[]]

        for i in range(0, len(Intervals)):

            currInter=Intervals[i]
            a1=currInter[1]-currInter[0]
            a2=currInter[2]-currInter[1]
            if a1<a2:
                next=currInter[2]-a1
                currInter = [currInter[0], currInter[1], next, currInter[2]]
            else: #a1>a2
                next=currInter[0]+a2
                currInter = [currInter[0], next, currInter[1], currInter[2]]

            Intervals[i]=currInter

            newCombi=[]
            for j in currInter:
                for k in combinations:
                    new=k.copy()
                    new.append(j)
                    newCombi.append(new)
            combinations=newCombi

        min = math.inf
        minCombi = []

        for combi in combinations:
            strCombi=str(combi)
            if strCombi not in valuesDict:
                valuesDict[strCombi]=function(combi)

            if valuesDict[strCombi]<min:
                min=valuesDict[strCombi]
                minCombi=combi

        for i in range(0, len(minCombi)):

            inter=Intervals[i]
            index=inter.index(minCombi[i])
            if index==0:
                index+=1
            elif index==3:
                index-=1
            Intervals[i]=[inter[index-1], inter[index], inter[index+1]]

        c=Intervals[0][2]-Intervals[0][0]


    ret={}

    for i in range(0, len(functionArgs)):

        ret[functionArgs[i]]=Intervals[i][1]

    return ret