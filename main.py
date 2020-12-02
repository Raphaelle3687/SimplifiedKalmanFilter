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
np.seterr("raise")

def testK():
    data=SDS("2018-2019", "H")
    d=data.data
    rk=Model("Thurstone", 600, args=[0.5])
    alpha=30
    beta=0.9992

    d.setVar0(alpha)
    inf1 = SKF(d, rk)
    inf1.infer(alpha, beta, iter=1)
    print(getLSOnInfer(inf1))

    for m in range(len(d.parametersMean[-1])):
        print(data.ItoPlayers[m])
        print(d.parametersMean[-1][m])

    #print(d.parametersVar[100])
    #print(d.parametersVar[-1])

def test():
    data=SDS("2015-2016", "H")
    d=data.data
    rk=Model("Thurstone", 600, args=[0.5])
    alpha=25
    beta=0.9994

    inf1 = SKF(d, rk)
    inf1.infer(alpha, beta, iter=1)
    print(getLSOnInfer(inf1))

    d.resetParam()

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
    beta=0.99
    alpha=1-beta**2

    model = Model("Thurstone", scale)

    X,Y,P,skills=genCompleteSynth(20, 200, alpha, beta, model=model)


    print(getMeanLS(X, Y, skills,model, P=P))

    data=DataSet(X, Y)
    data.setVar0(alpha)
    inf=SKF(data, model)
    inf.infer(alpha, beta)
    print(getLSOnInfer(inf, P=P))

    print(skills)
    print(data.parametersMean)
    print(data.parametersVar)

def createSyntheticDataSet(alpha, beta, model):
    X, Y, P, skills = genCompleteSynth(40, 200, alpha, beta, model=model)
    data=DataSet(X, Y)
    data.P=P
    return data

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
    betRange=np.arange(9990, 10001, 2)/10000

    #25, 0,9994

    #val=naiveOpti(LSon5, [alphRange, betRange])
    #print(val)

    plotLS3D(LSon5, alphRange, betRange, scale)



#optiOnHockey(iter=1)
testK()
#optiOn5()