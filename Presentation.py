import numpy as np
import matplotlib
#matplotlib.use("pdf")
import matplotlib.pyplot as plt
from Misc import *
from SKF import SKF
from KalmanFilter import KF
from Elo import Elo

def plotArgs(model, epsArgs, betArgs, mode, iter=1):
    if mode=="NHL":
        data=dataNHL
        K=K_H
    elif mode=="S1":
        data=dataS1
        K=K_S1
    elif mode=="S2":
        data=dataS2
        K=K_S2
    elif mode=="Gauss":
        data=dataGauss

    colors=["c", "m", "y", "r", "g", "b"]

    inferSKF=[]
    inferKF=[]

    EloLS=0

    for D in data:
        inferSKF.append(SKF(D, model))
        inferKF.append(KF(D, model))
    if mode!="Gauss":
        for D in data:
            elo=Elo(D, model.scale)
            elo.infer(K, 0)
            EloLS+=elo.getMeanLS(P=D.P)/len(data)


        plt.plot(epsArgs, np.ones(len(epsArgs))*EloLS, "k", label="Elo")

    def LSon5(params):
        eps=params[0]
        bet=params[1]
        temp1=0; temp2=0
        print(eps, bet)

        for inf in inferSKF:
            inf.data.resetParam()
            inf.infer(eps, bet, iter=iter)
            temp1+=getLSOnInfer(inf, P=inf.data.P)


        for inf2 in inferKF:
            inf2.data.resetParam()
            inf2.infer(eps, bet, iter=iter)
            temp2+=getLSOnInfer(inf2, P=inf2.data.P)

        return temp1/len(inferSKF), temp2/len(inferKF)

    for beta in betArgs:
        arraySKF = []
        arrayKF=[]
        for epsilon in epsArgs:
            skf, kf=LSon5([epsilon, beta])
            #skf= LSon5([epsilon, beta])
            arraySKF.append(skf)
            arrayKF.append(kf)

        col=colors.pop()
        plt.plot(epsArgs,arraySKF, color=col, label="beta: "+str(beta));
        plt.plot(epsArgs, arrayKF, color=col, linestyle="--");
    plt.legend(loc="upper left")
    ax=plt.axes();
    ax.set(yLabel="mean LS")
    ax.set(xLabel="epsilon")
    #plt.show()
    plt.savefig(mode+".png")
    plt.clf()



def plotLS3D(function, xArgs, yArgs, scale):

    V, S = np.meshgrid(xArgs, yArgs)
    res = []
    # print(V, S)
    for i in range(0, len(V)):
        res.append(list(zip(V[i], S[i])))
    Z = []
    for i in range(0, len(V)):
        Zi = []
        for j in range(0, len(S[i])):
            xij = V[i][j]
            yij = S[i][j]
            Zi.append(function([xij, yij]))
        Z.append(Zi)

    Z = np.array(Z)

    #minIndex=np.unravel_index(np.min(Z), Z.shape)
    #minIndex=np.where(Z == np.amax(Z))

    min=Z[0][0]
    minIndex=[0,0]
    for i in range(len(Z)):
        for j in range(len(Z[i])):
            if Z[i][j]<min:
                min=Z[i][j]
                minIndex=[i, j]


    print(V[minIndex[0]][minIndex[1]], S[minIndex[0]][minIndex[1]])

    ax = plt.axes(projection='3d')
    ax.plot_surface(V, S, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none');
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_title("sigma: "+str(scale))
    plt.show()





def plotLS(data, ax, args, mode, iter=1):

    interval=args[0]
    if mode=="NHL":
        model = Model("BradleyTerry", 1)
        epsilon = 5e-6
        beta = 1
    else:
        model=Model("Thurstone", 1)
        epsilon=6e-2
        beta=0.98

    KFs=[]
    SKFs=[]
    ELOs=[]
    for d in data:
        KFs.append(KF(d.copy(), model))
        SKFs.append(SKF(d.copy(), model))
        ELOs.append(Elo(d.copy(), 1))

    for i in range(len(data)):
        KFs[i].infer(epsilon, beta, iter=iter)
        SKFs[i].infer(epsilon, beta, iter=iter)
        ELOs[i].infer(K_H, beta)

    n=len(data[0].output)
    x=np.arange(0, n-(n%interval)+interval, interval)
    print(x)
    colors=["y", "m", "g", "b", "k"]
    yKF=np.zeros(len(x)-1)
    ySKF=np.zeros(len(x)-1)
    yELO=np.zeros(len(x)-1)
    for j in range(len(data)):
        for i in range(0, len(x)-1):
            yKF[i]+=(getLSOnInfer(KFs[j], start=x[i], end=x[i+1]))/len(data)
            ySKF[i]+=(getLSOnInfer(SKFs[j], start=x[i], end=x[i+1]))/len(data)
            yELO[i]+=(ELOs[j].getMeanLS(start=x[i], end=x[i + 1]))/len(data)

    ax.plot(x[1:], yKF, color=colors.pop(), label="KF")
    ax.plot(x[1:], ySKF, color=colors.pop(), label="SKF")
    ax.plot(x[1:], yELO, color=colors.pop(), label="Elo")

    ax.set(ylabel="LS")
    ax.set_title("Mean LS over time intervals, data:"+mode)
    ax.legend(loc="upper right")

def plotSkills(data, mode, ax):

    if mode=="NHL":
        epsilon=6e-5
        beta=1
    elif mode=="S1":
        epsilon = 0.02
        beta = 0.98
    elif mode=="S2":
        epsilon = 0.02
        beta = 0.98

    model = Model("Thurstone", 1)

    KFs=[]
    SKFs=[]
    ELOs=[]

    for d in data:
        KFs.append(KF(d.copy(), model))
        SKFs.append(SKF(d.copy(), model))
        ELOs.append(Elo(d.copy(), 1))

    for i in range(len(data)):
        KFs[i].infer(epsilon, beta)
        SKFs[i].infer(epsilon, beta)
        ELOs[i].infer(K_H, beta)

    n=len(data[0].output)
    interval=1
    x=np.arange(0, n-(n%interval)+interval, interval)
    print(x)
    colors=["y", "m", "g", "b", "k"]
    yKF=np.zeros(len(x)-1)
    ySKF=np.zeros(len(x)-1)
    yELO=np.zeros(len(x)-1)
    for j in range(len(data)):
        for i in range(0, len(x)-1):
            yKF[i]+=(getLSOnInfer(KFs[j], start=x[i], end=x[i+1]))/len(data)
            ySKF[i]+=(getLSOnInfer(SKFs[j], start=x[i], end=x[i+1]))/len(data)
            yELO[i]+=(ELOs[j].getMeanLS(start=x[i], end=x[i + 1]))/len(data)

    ax.plot(x[1:], yKF, color=colors.pop(), label="KF")
    ax.plot(x[1:], ySKF, color=colors.pop(), label="SKF")
    ax.plot(x[1:], yELO, color=colors.pop(), label="Elo")

    ax.set(ylabel="LS")
    ax.set_title("Mean LS over time intervals")
    ax.legend(loc="upper right")

