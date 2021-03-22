import math
import numpy.random as rand
import numpy as np
from Model import Model
import random
from DataSet import DataSet
from SportDataSet import SportDataSet
from Elo import Elo
from Trueskill import Trueskill
from Glicko import Glicko

def randomWalk(beta, epsilon,N, D, replacement=["rand", 50]):

    replace=False
    when=None
    if len(replacement)>0:
        replace=True
        toReplace=replacement[0]
        when=replacement[1]

    skills=np.zeros([D, N])
    I=np.identity(skills.shape[1])
    #COV=rand.random(I.shape)

    covMat=I*epsilon

    skills[0]=rand.multivariate_normal(skills[0], I)

    indexes=[]

    for i in range(D):
        if i==0:
            continue
        skills[i] = rand.multivariate_normal(beta * skills[i - 1], covMat)

        if replace==True and i==when:

            if toReplace=="top":

                med=np.quantile(skills[i], 0.5)
                indexes=np.where(skills[i]>med)[0]
                for j in indexes:
                    skills[i][j]=rand.normal(0, 1)

            elif toReplace=="bottom":
                med = np.quantile(skills[i], 0.5)
                indexes = np.where(skills[i] < med)[0]
                for j in indexes:
                    skills[i][j]=rand.normal(0, 1)
            else:
                n=int(N/2)
                indexA=np.arange(0, N).tolist()
                newSchedule = random.sample(indexA, len(indexA))
                for j in range(n):
                    skills[i][newSchedule[j]]=rand.normal(0, 1)
                indexes=newSchedule[:n]

    if replace==True:
        return skills, [when, indexes]
    else:
        return skills, []

def genSynthGaussian(N, D, epsilon, beta, scale=1):
    skills=randomWalk(beta, epsilon, N, D)
    X = np.zeros([D, N, N])
    Y = np.zeros([D, N, 1])
    P = np.zeros([D, N, 1])
    for i in range(D):
        for j in range(N):
            xij=np.zeros(N)
            xij[j]=1
            X[i][j] = xij
            z = np.dot(xij, skills[i])
            ech = rand.randn(1)
            yij = (ech + z) * scale
            #pij = norm.pdf(ech)
            Y[i][j] = yij
            P[i][j] = 1

    return X,Y,P,skills

def genSynthModel(M, D, epsilon, beta, model=Model("Thurstone", 1), replacement=[]):

    indexA=np.arange(0, M).tolist()

    skills, indexes=randomWalk(beta, epsilon,M, D, replacement=replacement)
    X=np.zeros([D,int(M/2), M])
    Y=np.zeros([D,int(M/2), model.yDim])
    P=np.zeros([D,int(M/2), model.yDim])

    for i in range(0, D):

        newSchedule=random.sample(indexA, len(indexA))

        for j in range(0, int(M/2)):

            xij=np.zeros(M)
            xij[newSchedule.pop()]=1
            xij[newSchedule.pop()]=-1
            pij=np.zeros(model.yDim)

            for k, f in enumerate(model.yFunctions):
                pij[k]=  f (skills[i], xij)

            yij=np.zeros(model.yDim)
            yij[biasedDice(pij)]=1
            X[i][j]=xij
            P[i][j]=pij
            Y[i][j]=yij

    return X,Y,P,skills, indexes


def biasedDice(bias):
    x=rand.rand(1)
    sum=0
    for i, val in enumerate(bias):
        if x<sum+val:
            return i
        else:
            sum+=val

def createSyntheticDataSet(alpha, beta, model, players=10, days=100, variant="", replacement=[]):
    if variant=="gaussian":
        X, Y, P, skills = genSynthGaussian(players, days, alpha, beta, scale=model.scale)
    else:
        X, Y, P, skills, indexes = genSynthModel(players, days, alpha, beta, model=model, replacement=replacement)
    data=DataSet(X, Y, X.shape[2], Y.shape[2])
    data.P=P
    return data, indexes

seasons_K=["2007-2008", "2008-2009", "2009-2010", "2010-2011", "2011-2012"]
dataNHL_K=[]
dataS1_K=[]
dataS2_K=[]

seasons=["2013-2014", "2014-2015", "2015-2016", "2016-2017", "2017-2018"]
dataNHL=[]
dataS1=[]
dataS2=[]
dataGauss=[]

mod=Model("Thurstone", 1)
beta=0.98
epsilon=(1-beta**2)
for i in range(30):
    if False:
        dataS1_K.append(createSyntheticDataSet(epsilon, beta, mod, players=4, days=200))
        dataS2_K.append(createSyntheticDataSet(epsilon, beta, mod, players=30, days=200))

for i in range(5):
    if False:
        dataNHL_K.append(SportDataSet(seasons_K[i], "H").data)
        dataS2_K.append(createSyntheticDataSet(epsilon, beta, mod, players=60, days=200))
    if False:
        dataNHL.append(SportDataSet(seasons[i], "H").data)
        #dataS1.append(createSyntheticDataSet(epsilon, beta, mod, players=12, days=200))
        #dataS2.append(createSyntheticDataSet(epsilon, beta, mod, players=60, days=200))
        #dataGauss.append(createSyntheticDataSet(epsilon, beta,mod, players=5, days=1000, variant="gaussian"))

#K_H=0.0081
K_H=0.0062
K_S1=0.0850
K_S2=0.0867


def getLSOnInfer(infer, P=None, start=None, end=None):

    paramM=infer.data.parametersMean
    paramV=infer.data.parametersVar
    input=infer.data.input
    output=infer.data.output

    beta=1
    model=None
    if isinstance(infer, Elo) or isinstance(infer, Trueskill) or isinstance(infer, Glicko):
        model=infer
        if not isinstance(infer, Elo):
            beta=model.beta
    else:
        model=infer.model
        beta=infer.beta



    return getMeanLS(input, output, paramM,model, paramVar=paramV,start=start, end=end, P=P, beta=beta)


def getMeanLS(input, output, paramMean, model, paramVar=None, start=None, end=None , P=None, beta=1):


    LS2 = 0
    if start == None:
        start = 0
    if end == None:
        end = len(input)

    if P is None:
        P = output

    realDays=0
    count = 0
    for i in range(start, end):
        Xi =input[i]
        if len(Xi)>0:
            realDays+=1

        LS1 = 0
        for j, xij in enumerate(Xi):
            count+=1
            param=paramMean[i]*beta
            probs=model.getProbs(param, xij, paramVar[i], output[i][j])
            for k, p in enumerate(probs):
                pReal=P[i][j][k]
                LS1 -=  pReal* math.log(p)

        LS2+=LS1

    #return LS2 / realDays
    return LS2/count


def naiveOpti(function, intervals):

    combinations=[[]]
    values=[]

    for inter in intervals:
        newCombinations=[]
        for comb in combinations:
            for val in inter:

                newComb=comb.copy()
                newComb.append(val)
                newCombinations.append(newComb)

        combinations=newCombinations

    for comb in combinations:
        values.append(function(comb))

    return combinations[values.index(min(values))]


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