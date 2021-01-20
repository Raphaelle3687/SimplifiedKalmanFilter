from SportDataSet import SportDataSet as SDS
from Model import Model
from SKF import SKF
from DataSet import DataSet
import numpy as np
import numpy.random as rand
import random
from Misc import *
from Presentation import *
from KalmanFilter import KF
from Elo import Elo
from datetime import date
from Trueskill import Trueskill
np.seterr("raise")
import plotly
from Glicko import Glicko

def testK():
    data=SDS("2018-2019", "H")
    d=data.data
    rk=Model("Thurstone", 1, args=[0.5])
    alpha=20/600**2
    beta=1

    d.setVar0(alpha)
    inf1 = KF(d, rk)
    inf1.infer(alpha, beta, iter=1)
    print(getLSOnInfer(inf1))

    for m in range(len(d.parametersMean[-1])):
        print(data.ItoPlayers[m])
        print(d.parametersMean[-1][m])

    #print(d.parametersVar[100])
    print(d.parametersVar[-1])

def test():
    data=SDS("2015-2016", "H")
    d=data.data
    rk=Model("Thurstone", 600, args=[0.5])
    alpha=20
    beta=0.9994

    inf1 = SKF(d, rk)
    inf1.infer(alpha, beta, iter=1)
    print(getLSOnInfer(inf1))

    d.resetParam()

    elo=Elo(d, 600)
    elo.infer(3000)
    print(elo.getMeanLS())

    for m in range(len(d.parametersMean[-1])):
        print(data.ItoPlayers[m])
        print(d.parametersMean[-1][m])

def test2D():
    data = SDS("2018-2019", "H")
    d = data.data
    rk = Model("RaoKupperGaussian", 600)
    alpha = 60
    beta = 1
    dSynth = d.generateSynthetic(alpha, rk)
    # d.output=dSynth.output

    infSynth = SKF(dSynth, rk)
    print(infSynth.getMeanLS())

    dSynthToInfer = DataSet(dSynth.input, dSynth.output)
    dSynthToInfer.setVar0(1)
    infSynth2 = SKF(dSynthToInfer, rk)
    infSynth2.infer(beta, alpha, iter=1)
    print(infSynth2.getMeanLS())

    for m in range(len(d.parametersMean[-1])):
        print(data.ItoPlayers[m])
        print(dSynthToInfer.parametersMean[-1][m], dSynth.parametersMean[-1][m])


def test3():

    scale=1
    beta=0.98
    alpha=1-beta**2

    model = Model("BradleyTerry", scale)

    X,Y,P,skills=genSynthModel(8, 1000, alpha, beta, model=model)

    #print(skills)
    #print(np.mean(skills, axis=0))
    #print(np.var(skills, axis=0))

    #print(getMeanLS(X, Y, skills,model, P=P))

    #data=DataSet(X, Y, 8, 2)
    data=dataNHL[0]

    inf=SKF(data, model)
    inf.infer(1.7e-4, 1)
    print(getLSOnInfer(inf, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))

    inf2=KF(data, model)
    inf2.data.resetParam()
    inf2.infer(1.1e-4, 1)
    print(getLSOnInfer(inf2, P=data.P))
    var=data.parametersVar[-1]
    var=np.diag(var)

    inf3=Trueskill(data, 1)
    inf3.data.resetParam()
    inf3.infer(3.3e-4, 1)
    print(getLSOnInfer(inf3, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))

    inf4=Glicko(data, 1)
    inf4.data.resetParam()
    inf4.infer(1.5e-4, 1)
    print(getLSOnInfer(inf4, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))

def testElo():
    scale=1
    beta=0.99
    alpha=1-beta**2

    model = Model("Thurstone", scale)

    X,Y,P,skills=genSynthModel(10, 100, alpha, beta, model=model)
    print(getMeanLS(X, Y, skills,model, P=P))

    data=DataSet(X, Y)
    inf=Elo(data, scale)
    inf.infer(5e-2)
    print(inf.getMeanLS(P=P))

def testGauss():

    scale=1
    beta=0.98
    alpha=1-beta**2

    model = Model("Thurstone", scale)

    X, Y, P, skills=genSynthGaussian(5, 100, alpha, beta)

    Y1=Y[:, 0, 0]
    S1=skills[:, 0]

    x=np.arange(100)
    plt.scatter(x, Y1, color="r")
    plt.plot(x, S1)
    plt.show()




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



def optiOn5(iter=1):
    scale = 10
    beta = 0.99
    alpha = (1 - beta ** 2)*100

    print("genValues:", alpha, beta)

    model = Model("Thurstone", scale)

    infers=[]

    for i in range(0, 5):
        data=createSyntheticDataSet(alpha, beta, model)
        infers.append(SKF(data, model))

    infTest=SKF(createSyntheticDataSet(alpha, beta, model), model)

    def LSon5(params):
        alph=params[0]
        bet=params[1]
        temp=0
        print(alph, bet)
        for inf in infers:
            inf.data.resetParam()
            inf.infer(alph, bet, iter=iter)
            temp+=getLSOnInfer(inf, P=inf.data.P)

        ret=temp/len(infers)
        print(ret)

        return ret


    alphRange=np.arange(0.5,5.1, 0.5)
    betRange=np.arange(9900, 10001, 20)/10000
    val=plotLS3D(LSon5, alphRange, betRange, scale)

    infTest.infer(val["alpha"], val["beta"], iter=iter)
    print(getLSOnInfer(infTest))

def optiOnHockey(iter=1):

    scale=100

    model=Model("Thurstone", scale)

    seasons=["2007-2008","2008-2009","2009-2010", "2010-2011","2011-2012"]
    #seasons=["2007-2008"]

    infers=[]

    for s in seasons:
        data=SDS(s, "H")
        infers.append(KF(data.data, model))

    def LSon5(params):
        alph=params[0]
        bet=params[1]
        temp=0
        print(alph, bet)
        for inf in infers:
            inf.data.resetParam()
            inf.infer(alph, bet, iter=iter)
            temp+=getLSOnInfer(inf, P=inf.data.P)

        ret=temp/len(infers)
        print(ret)

        return ret

    #alphRange=np.arange(1,101, 20)
    #betRange=np.arange(999, 1000, 0.2)/1000
    alphRange=np.arange(0,2.1, 0.2)
    betRange=np.arange(10000, 10001, 2)/10000

    #25, 0,9994

    #val=naiveOpti(LSon5, [alphRange, betRange])
    #print(val)

    plotLS3D(LSon5, alphRange, betRange, scale)


def plot():

    scale = 1
    model = Model("Thurstone", scale)
    modelNHL=Model("BradleyTerry", scale)
    modelGauss=Model("Gaussian", 1)
    epsRange1=np.arange(0.1, 10.2, 2)/10000
    betRange1 = [1]
    epsRange2=np.arange(0.1, 10.1, 1)/100
    betRange2=[0.95, 0.98, 1]
    betRange2=[0.98]
    epsRange3 = np.arange(0.2, 10.1, 0.2)/100
    betRange3 = [0.95, 0.98, 1]
    epsRange4=np.arange(2, 8.1, 0.5)/100
    betRange4=[0.95, 0.98, 1]

    #plotArgs(modelGauss, epsRange4, betRange4, "Gauss")
    plotArgs(modelNHL, epsRange1, betRange1, "NHL")
    #plotArgs(model, epsRange2, betRange2, "S1")
    #plotArgs(model, epsRange3, betRange3, "S2")

def plot2():
    fig, ax=plt.subplots()
    plotLS(dataNHL, ax, [20], mode="NHL")
    #plt.show()
    plt.savefig("LSoverTimeNHL.png")


def findVar(data,scale, eps1, bet1, eps2, bet2, eps3, bet3, eps4, bet4):

    model = Model("BradleyTerry", scale)

    inf=SKF(data, model)
    inf.infer(eps1, bet1, var0=70)
    print(getLSOnInfer(inf, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))

    mean=data.parametersMean
    y=np.var(mean,axis=1)
    x=np.arange(0, len(y))
    plt.plot(x, y)


    inf2=KF(data, model)
    inf2.data.resetParam()
    inf2.infer(eps2, bet2, var0=130)
    print(getLSOnInfer(inf2, P=data.P))
    var=data.parametersVar[-1]
    var=np.diag(var)
    print(np.mean(var))

    mean=data.parametersMean
    y2=np.var(mean,axis=1)
    plt.plot(x, y2)
    plt.show()

    inf3=Trueskill(data, scale)
    inf3.data.resetParam()
    inf3.infer(eps3, bet3)
    print(getLSOnInfer(inf3, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))

    inf4=Glicko(data, scale)
    inf4.data.resetParam()
    inf4.infer(eps4, bet4)
    print(getLSOnInfer(inf4, P=data.P))
    var = data.parametersVar[-1]
    print(np.mean(var))


def createInf(data, name, scale):
    if name=="SKF-BT":
        model=Model("BradleyTerry", scale)
        return SKF(data, model)
    elif name=="SKF-T":
        model=Model("Thurstone", scale)
        return SKF(data, model)
    elif name=="KF-BT":
        model=Model("BradleyTerry", scale)
        return KF(data, model)
    elif name=="KF-T":
        model=Model("Thurstone", scale)
        return KF(data, model)
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


def getTableLSEpsilon(data):

    epsArgsH=np.arange(0.1, 4.2, 0.4)*100
    betArgsH=[0.98]
    var0H=[0]

    scale=100
    K=K_H*scale

    #names= ["SKF","KF","Trueskill","Glicko", "Elo"]
    names=["SKF-BT", "KF-BT", "Glicko"]
    Infers=getInfers(data,names, scale)
    epsArgs = []
    betArgs=[]
    varArgs=[]

    for i in range(len(names)):
        epsArgs.append(epsArgsH)
        betArgs.append(betArgsH)
        varArgs.append(var0H)

    plotArgs(Infers, epsArgs, betArgs, varArgs, K, "TestS1")

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


def getTableLSVar(data,eps,var,scale, name):

    K=K_H*scale**2

    epsValues=np.arange(0, 1.1, 0.2)
    varValues=np.arange(0.1, 1.2, 0.2)


    names= [name]
    #names=["Trueskill", "Elo"]
    Infers=getInfers(data,names, scale)
    epsArgs = [epsValues*eps]
    betArgs=[[1]]
    varArgs=[varValues*var]

    plotArgs(Infers, epsArgs, betArgs, varArgs, K, name)




#optiOnHockey(iter=1)
#testK()
#optiOn5()
#plot2()
#test3()
#testElo()
#test()
#optiElo_K()
#plot()
#testGauss()
#findVar(dataNHL_K[0], 100, 0.9, 1, 0.9, 1, 0.9, 1, 0.9, 1 )
getTableLSEpsilon(dataS2_K)
scale=100
#getTableLSVar(dataNHL_K,0.9,150,scale, "SKF-BT")
#getTableLSVar(dataNHL_K,0.9,150,scale, "KF-BT")
#getTableLSVar(dataNHL_K,0.9,150,scale, "Glicko")
#getTableLSVar(dataNHL_K,1.7,290,scale, "SKF-T")
#getTableLSVar(dataNHL_K,1.3,240,scale, "KF-T")
#getTableLSVar(dataNHL_K,1.7,300,scale, "Trueskill")
#getLSTableNHL()
