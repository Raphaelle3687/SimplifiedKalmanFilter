from SportDataSet import SportDataSet as SDS
from Model import Model
from SKF import SKF
from DataSet import DataSet
import numpy as np
import numpy.random as rand
from Misc import *
from raiting_algorithms import *
import pandas as pd
from Presentation import *
from KF import KF
from KalmanFilter import KalmanFilter
from Elo import Elo
from Trueskill import Trueskill
np.seterr("raise")
from time import time
from Glicko import Glicko as G
from scipy.stats._continuous_distns import _norm_pdf as pdf
from scipy.stats._continuous_distns import _norm_cdf as cdf
from scipy.stats import norm
from scipy.stats._continuous_distns import _norm_pdf
from joblib import Parallel, delayed
#matplotlib.use("pdf")

def optiElo_K():

    allData=[dataNHL_K, dataS1_K, dataS2_K]

    for data in allData:

        models=[]
        for dataPoint in data:
            models.append(Elo(dataPoint, 1))

        def costFunc(K):
            temp=0
            K=K[0]
            for mod in models:
                mod.data.resetParam()
                mod.infer(K, 0)
                temp+=getLSOnInfer(mod, start=int(len(mod.data.output)/2),P=mod.data.P)
            val=temp/len(models)
            print(K, val)
            return val

        K=goldenSearch(costFunc, ["K"], [[0, 1]], reduction=0.001)
        print(K)

def plot_e_v0(iter=1):
    scale = 1
    beta = 0.98
    alpha = (1 - beta ** 2)*scale**2

    print("genValues:", alpha, beta)

    model = Model("Thurstone", scale)

    data=dataNHL_K

    infers=[]
    n=30
    for i in range(0, n):
        data=createSyntheticDataSet(alpha, beta, model, players=8, days=100)
        infers.append(KalmanFilter(data, model, "KF"))

    def LSon5(params):
        e=params[0]
        v0=params[1]
        temp=0
        print(e, v0)
        for inf in infers:
            inf.data.resetParam()
            inf.infer(e, 0.98, var0=v0, iter=iter)
            temp+=getLSOnInfer(inf, P=inf.data.P)

        ret=temp/len(infers)
        print(ret)

        return ret


    epsRange = alpha*np.arange(30,171, 10)/100
    #betRange=np.arange(9900, 10001, 50)/10000
    v0Range = scale*np.arange(30, 171, 10)/100
    val=plotLS3D(LSon5, epsRange, v0Range, scale)

    print(val)

def createInf(data, name, scale):
    if name=="vSKF-BT":
        model=Model("BradleyTerry", scale)
        return KalmanFilter(data, model, "vSKF")
    elif name=="vSKF-T":
        model=Model("Thurstone", scale)
        return KalmanFilter(data, model, "vSKF")
    elif name=="KF-BT":
        model=Model("BradleyTerry", scale)
        return KalmanFilter(data, model, "KF")
    elif name=="KF-T":
        model=Model("Thurstone", scale)
        return KalmanFilter(data, model, "KF")
    elif name=="sSKF-BT":
        model=Model("BradleyTerry", scale)
        return KalmanFilter(data, model, "sSKF")
    elif name=="sSKF-T":
        model=Model("Thurstone", scale)
        return KalmanFilter(data, model, "sSKF")
    elif name=="fSKF-BT":
        model=Model("BradleyTerry", scale)
        return KalmanFilter(data, model, "fSKF")
    elif name=="fSKF-T":
        model=Model("Thurstone", scale)
        return KalmanFilter(data, model, "fSKF")
    elif name=="Elo":
        return Elo(data, scale)
    elif name=="Glicko":
        return Glicko(data, scale)
    elif name=="Trueskill":
        return Trueskill(data, scale)

def getInfers(data, list, scale):

    Infers=[]

    for name in list:
        infers=[]
        for D in data:
            infers.append(createInf(D, name, scale))
        Infers.append(infers)
    return Infers

def getLSTableNHL():
    scale=100
    K=K_H*scale**2

    #names= ["SKF","KF","Trueskill","Glicko", "Elo"]
    names=["SKF-T","KF-T","Trueskill", "SKF-BT", "KF-BT", "Glicko", "Elo"]
    Infers=getInfers(dataNHL,names, scale)
    epsArgs = [[1.36], [1.04], [1.36], [0.72], [0.54], [0.72], [0]]
    betArgs=[[1],[1],[1],[1],[1],[1],[1]]
    varArgs=[[29], [24], [30], [15], [15], [15], [0]]
    plotArgs(Infers, epsArgs, betArgs, varArgs, K, "LS_NHL_Opti")

def optimalSearchA(algoList, dataList, epsilon, beta, scale, iter=1):

    timeInLS=0
    timeInKF=0

    Algos=getInfers(dataList, algoList, scale)
    epsRange=epsilon*np.array([1])
    betRange=[0.98, beta, 1]
    varRange=[0, 1, 2]

    n=len(dataList)

    minLS={}
    minParam={}
    for name in algoList:
        minLS[name]=10
        minParam[name]=None

    I=0
    for e in epsRange:
        for b in betRange:
            for v0 in varRange:
                I+=1
                print(I, "e:"+str(e), "b:"+str(b), "v0:"+str(v0))
                for i, name in enumerate(algoList):
                    LS=0
                    for Ai in Algos[i]:
                        Ai.data.resetParam()
                        a=time()
                        Ai.infer(e, b, var0=v0, iter=iter)
                        B=time()
                        timeInKF+=(B-a)
                        c=time()
                        LS+=getLSOnInfer(Ai, P=Ai.data.P, start=int(len(Ai.data.input)/2))/n
                        d=time()
                        timeInLS+=(d-c)
                    if LS<minLS[name]:
                        minLS[name]=LS
                        minParam[name]={"e":e, "b":b, "v0":v0}

    print(timeInKF)
    print(timeInLS)
    return minLS, minParam

def getProbs(data, model, scale,e, b, v0, iter, indexes):
    A=createInf(data, model, scale)
    return A.infer(e, b, var0=v0, iter=iter, switchIndex=indexes)

def genData(n, e, b, model=Model("Thurstone", 1), replacement=[]):
    dataList=[]
    indexes=[]
    for i in range(n):
        data, index=createSyntheticDataSet(e, b,model, 20, 100, replacement=replacement)
        dataList.append(data)
        indexes.append(index)
    return dataList, indexes

def meanLS(probs, P):
    LS=[]
    for n in range(len(probs)):
        LSn=[]
        for i in range(len(probs[n])):
            LSn.append(np.sum(-np.log(probs[n][i])*P[n][i]))
        LS.append(LSn)

    return LS

def D_KL(probs, P):

    DK=[]
    for n in range(len(probs)):
        DKn=[]
        for i in range(len(probs[n])):
            DKn.append(np.sum( P[n][i] * np.log(P[n][i]/probs[n][i]) ))
        DK.append(DKn)
    return DK


def Figure1(b=0.998, s=1, v0=1, dkl=True, replacement=[]):
    replace=""
    if len(replacement)>0:
        replace=replacement[0]

    a=time()

    scale=s
    beta=b
    var0=v0
    epsilon=1-beta**2
    model=Model("Thurstone", scale)
    N=5000
    algos=["KF-T"]

    betaHat=[0.98, 0.998, 1]
    #betaHat=[0.998]

    dataList, indexes=genData(N, epsilon, beta, model=model, replacement=replacement)

    x=np.arange(0, len(dataList[0].input))
    A=algos[0]

    colors=["m", "g", "b"]


    for bH in betaHat:

        probA=Parallel(n_jobs=4)(delayed(getProbs)(D, A, 1,  epsilon, bH, var0, 1, indexes[i]) for i, D in enumerate(dataList))
        if dkl==True:
            plot = Parallel(n_jobs=4)(delayed(D_KL)(probA[i], D.P) for i, D in enumerate(dataList))
        else:
            plot=Parallel(n_jobs=4)(delayed(meanLS)(probA[i], D.P) for i,D in enumerate(dataList))

        #for i, D in enumerate(dataList):
        #    probA[i]=np.mean(meanLS(probA[i], D.P), axis=1)


        c=colors.pop()
        plt.plot(x, np.mean(plot, axis=(0, 2)), color=c,linestyle="solid",label=bH)
        quant3=np.quantile(plot, 0.75,  axis=(2, 0))
        plt.plot(x, quant3,color=c,linestyle="dashed")
        med=np.quantile(plot, 0.5,  axis=(2, 0))
        plt.plot(x, med, color=c, linestyle="dotted")

    b=time()
    print(b-a)
    plt.legend(loc="upper right")
    plt.title("beta:"+str(beta)+"  sigma:"+str(scale)+"  var0:"+str(var0))
    plt.show()
    if dkl:
        name="--D_KLoverTime-"
    else:
        name="--LSoverTime-"
    plt.savefig("fig/"+replace+"--"+str(A)+name+str(beta)+"-sig-"+str(scale)+"-v0-"+str(var0)+".jpg")
    plt.clf()

def Figure2(b=0.998, s=1, v0=1, dkl=True, replacement=[]):
    replace=""
    if len(replacement)>0:
        replace=replacement[0]

    a=time()

    scale=s
    beta=b
    var0=v0
    epsilon=1-beta**2
    bHat = 1
    model=Model("Thurstone", scale)
    N=1000
    algos=["vSKF-T","Trueskill","Glicko"]
    vars=[1, 1, 1]

    dataList, indexes=genData(N, epsilon, beta, model=model, replacement=replacement)

    x=np.arange(0, len(dataList[0].input))
    A=algos[0]

    colors=["m", "g", "b", "k", "y","r"]
    plt.figure(figsize=(15, 8))


    for k, A in enumerate(algos):

        probA=Parallel(n_jobs=4)(delayed(getProbs)(D, A, 1,  epsilon, bHat, vars[k], 1, indexes[i]) for i, D in enumerate(dataList))
        if dkl==True:
            plot = Parallel(n_jobs=4)(delayed(D_KL)(probA[i], D.P) for i, D in enumerate(dataList))
        else:
            plot=Parallel(n_jobs=4)(delayed(meanLS)(probA[i], D.P) for i,D in enumerate(dataList))

        #for i, D in enumerate(dataList):
        #    probA[i]=np.mean(meanLS(probA[i], D.P), axis=1)


        c=colors.pop()
        plt.plot(x, np.mean(plot, axis=(0, 2)), color=c,linestyle="solid",label=algos[k])
        #quant3=np.quantile(plot, 0.75,  axis=(2, 0))
        #plt.plot(x, quant3,color=c,linestyle="dashed")
        med=np.quantile(plot, 0.5,  axis=(2, 0))
        plt.plot(x, med, color=c, linestyle="dotted")

    b=time()
    print(b-a)
    plt.legend(loc="upper right")
    plt.title("beta:"+str(beta)+"  sigma:"+str(scale)+"  var0:"+str(var0))
    plt.ylim([0, 0.3])
    plt.show()
    if dkl:
        name="--D_KLoverTime-"
    else:
        name="--LSoverTime-"
    plt.savefig("fig/"+replace+"--"+"vSFK_TS_GL"+name+str(beta)+"-sig-"+str(scale)+"-v0-"+str(var0)+".jpg")
    plt.clf()

def Figure3(b=0.998, s=1, v0=1, dkl=True, replacement=[]):
    replace=""
    if len(replacement)>0:
        replace=replacement[0]

    a=time()

    scale=s
    beta=b
    var0=v0
    epsilon=1-beta**2
    bHat = 1
    model=Model("Thurstone", scale)
    N=1000
    algos=["fSKF-T", "fSKF-BT","Elo"]
    vars=[0.05, 0.1, 0.2, 0.3]

    dataList, indexes=genData(N, epsilon, beta, model=model, replacement=replacement)

    x=np.arange(0, len(dataList[0].input))
    A=algos[0]

    colors=["m", "g", "b", "k", "y","r"]
    linestyles=["solid", "dashed", "dotted", "dashdot"]
    plt.figure(figsize=(15, 8))


    for kk, A in enumerate(algos):

        c = colors.pop()

        for k in range(len(vars)):

            probA=Parallel(n_jobs=4)(delayed(getProbs)(D, A, 1,  epsilon, bHat, vars[k], 1, indexes[i]) for i, D in enumerate(dataList))
            if dkl==True:
                plot = Parallel(n_jobs=4)(delayed(D_KL)(probA[i], D.P) for i, D in enumerate(dataList))
            else:
                plot=Parallel(n_jobs=4)(delayed(meanLS)(probA[i], D.P) for i,D in enumerate(dataList))

        #for i, D in enumerate(dataList):
        #    probA[i]=np.mean(meanLS(probA[i], D.P), axis=1)



            plt.plot(x, np.mean(plot, axis=(0, 2)), color=c,linestyle=linestyles[k],label=algos[kk]+"v0:"+str(vars[k]))
        #quant3=np.quantile(plot, 0.75,  axis=(2, 0))
        #plt.plot(x, quant3,color=c,linestyle="dashed")
        #med=np.quantile(plot, 0.5,  axis=(2, 0))
        #plt.plot(x, med, color=c, linestyle="dotted")

    b=time()
    print(b-a)
    plt.legend(loc="upper right")
    plt.title("beta:"+str(beta)+"  sigma:"+str(scale))
    plt.ylim([0, 0.5])
    plt.show()
    if dkl:
        name="--D_KLoverTime-"
    else:
        name="--LSoverTime-"
    plt.savefig("fig/"+replace+"--"+"fSKFvsELO"+name+str(beta)+"-sig-"+str(scale)+"-v0-"+str(var0)+".jpg")
    plt.clf()


def plotSkills():

    scale=1
    beta=0.98
    epsilon=1-beta**2
    model=Model("Thurstone", scale)

    X, Y, P, skills = genSynthModel(8, 100, epsilon, beta, model=model)
    #varSkills=np.var(skills, axis=1)
    x=np.arange(0,len(skills))
    plt.plot(x, skills)
    plt.show()

def translate(data):

    Y=data.output
    X=data.input
    P=data.P
    col=["home_player", "away_player", "game_result", "real_proba", "time_stamp"]
    D=[]
    num=np.arange(0, X.shape[2])
    for n in range(len(X)):
        for i in range(len(X[n])):
            home=np.where(X[n, i]==1)[0][0]
            away=np.where(X[n, i]==-1)[0][0]
            result=np.where(Y[n, i]==0)[0][0]
            result=Y[n, i][0]
            realProb=P[n, i][0]
            timestamp=n

            D.append([home, away, result, realProb, timestamp])

    return pd.DataFrame(D, columns=col)

def test():

    scale=1
    beta=0.998
    epsilon=1-beta**2
    model=Model("Thurstone", scale)

    X, Y, P, skills, indexes = genSynthModel(10, 200, epsilon, beta, model=model)
    data=DataSet(X, Y, X.shape[2], Y.shape[2])
    data.P=P
    res=translate(data)

    beta=1

    PAR={"home":0, "beta":beta, "epsilon":epsilon, "v0":1, "scale":scale, "rating_algorithm":"Glicko","rating_model":"Bradley-Terry","it":1,"metric_type":"DKL", "PAR_gen":{"scenario":None}}

    skills, LS, V, MSE=Glickog(res, PAR)
    #print(skills.loc[0])

    KF1=G(data.copy(), scale)
    probs=KF1.infer(epsilon, beta, var0=scale-epsilon, iter=1)
    V2=KF1.data.parametersVar
    S=KF1.data.parametersMean
    DKL=D_KL(probs, P)

    LS=LS.reshape(200, 5)
    DKL=np.mean(DKL, axis=1)
    LS = np.mean(LS, axis=1)

    x=np.arange(1, 200)
    """plt.plot(x, DKL)
    plt.plot(x, LS)
    plt.show()"""

    skillsDiff=[]

    for i in range(len(KF1.data.parametersMean)-1):
        iT=5*i+4
        gneuh=skills.iloc[i]
        skillsDiff.append(KF1.data.parametersMean[i+1]-skills.iloc[iT])

    print(skillsDiff)

    plt.plot(x, skillsDiff)
    plt.show()

test()
#plotSkills()
#Figure2(0.998, 1, 1, replacement=["top", 50])
#Figure2(0.998, 1, 1, replacement=["bottom", 50])
#Figure2(0.998, 1, 1, replacement=["rand", 50])
#Figure3(0.998, 1, 1, replacement=["rand", 50])
#Figure2(0.998, 1, 1, replacement=[])
"""Figure1(0.998, 1, 1, replacement=["top", 50])
Figure1(0.998, 1, 1, replacement=["bottom", 50])
Figure1(0.998, 1, 1, replacement=["rand", 50])
Figure1(0.998, 1, 1, replacement=[])
Figure1(0.998, 1, 1,dkl=False, replacement=["top", 50])
Figure1(0.998, 1, 1,dkl=False, replacement=["bottom", 50])
Figure1(0.998, 1, 1,dkl=False, replacement=["rand", 50])
Figure1(0.998, 1, 1,dkl=False, replacement=[])"""
#Figure1(0.998, 1, 0)
#Figure1(0.98, 1, 1)
#Figure1(0.98, 1, 0)






