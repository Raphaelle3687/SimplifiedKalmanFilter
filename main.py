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
np.seterr("raise")

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

    print(skills)
    print(np.mean(skills, axis=0))
    print(np.var(skills, axis=0))

    #print(getMeanLS(X, Y, skills,model, P=P))

    data=DataSet(X, Y)
    data.setVar0(alpha)
    inf=SKF(data, model)
    inf.infer(alpha, beta, iter=1)
    #print(getLSOnInfer(inf, P=P))
    #print(inf.data.parametersMean)

    inf2=KF(data, model)
    inf2.data.resetParam()
    inf2.infer(alpha, beta, iter=1)
    #print(getLSOnInfer(inf2, P=P))
    #print(inf2.data.parametersMean)

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
                temp+=mod.getMeanLS(P=mod.data.P)
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
    epsRange1=np.arange(0.1,5.2, 1)/100000
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
    plotLS(dataNHL, ax, [100], mode="NHL")
    plt.show()
    plt.savefig("LSoverTimeNHL.png")

#optiOnHockey(iter=1)
#testK()
#optiOn5()
#plot2()
#test3()
#testElo()
#test()
#optiElo_K()
plot()
#testGauss()