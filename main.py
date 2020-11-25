from SportDataSet import SportDataSet as SDS
from Model import Model
from Infer import Infer
from DataSet import DataSet
import numpy as np
import numpy.random as rand
import random
from Misc import *
np.seterr("raise")

def test():
    data=SDS("2018-2019", "H")
    d=data.data
    rk=Model("Thurstone", 600, args=[0.5])
    alpha=60
    beta=0.99

    d.setVar0(1)
    inf1 = Infer(d, rk)
    inf1.infer(alpha, beta, iter=1)
    print(getLSOnInfer(inf1))

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

    infSynth = Infer(dSynth, rk)
    print(infSynth.getMeanLS())

    dSynthToInfer = DataSet(dSynth.input, dSynth.output)
    dSynthToInfer.setVar0(1)
    infSynth2 = Infer(dSynthToInfer, rk)
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
    inf=Infer(data, model)
    inf.infer(alpha, beta)
    print(getLSOnInfer(inf, P=P))

    print(skills)
    print(data.parametersMean)
    print(data.parametersVar)

def createSyntheticDataSet(alpha, beta, model):
    X, Y, P, skills = genCompleteSynth(200, 100, alpha, beta, model=model)
    data=DataSet(X, Y)
    data.P=P
    return data

def optiOn5(iter=1):
    scale = 0.5
    alpha = 0.2
    beta = 0.95
    model = Model("Thurstone", scale)

    infers=[]

    for i in range(0, 5):
        data=createSyntheticDataSet(alpha, beta, model)
        infers.append(Infer(data, model))

    infTest=Infer(createSyntheticDataSet(alpha, beta, model), model)

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

    val=goldenSearch(LSon5, ["alpha","beta"], [[0, 10],[0.95, 1]])
    print(val)

    infTest.infer(val["alpha"], val["beta"], iter=iter)
    print(getLSOnInfer(infTest))

def optiOnHockey(iter=1):

    model=Model("Thurstone", 10)

    seasons=["2007-2008","2008-2009","2009-2010", "2010-2011","2011-2012"]

    infers=[]

    for s in seasons:
        data=SDS(s, "H")
        infers.append(Infer(data.data, model))

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

    val=goldenSearch(LSon5, ["alpha","beta"], [[0.1, 10],[0.98, 1]])
    print(val)

#optiOnHockey(iter=1)
test3()