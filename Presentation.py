import numpy as np
import matplotlib
#matplotlib.use("pdf")
import matplotlib.pyplot as plt
from Misc import *
from SKF import SKF
from KF import KF
from Elo import Elo
from Trueskill import Trueskill
from Glicko import Glicko
import plotly.graph_objects as plotly
import matplotlib.ticker as mticker
from KalmanFilter import KalmanFilter

def getColorGrad(scores, mode="max"):
    minScore = min(scores)
    maxScore = max(scores)
    greenMin = 255
    redMin = 0

    sortedScores = scores.copy()
    sortedScores.sort()

    interval = int(255 / (len(scores)))

    colors = []

    if mode=="total":

        for i in scores:
            ind = sortedScores.index(i)
            colors.append("rgb(" + str(redMin + interval * ind) + "," + str(greenMin - interval * ind) + ", 0)")
    else:
        for i in scores:
            colors.append("w")
    if mode=="max":
        i = scores.index(minScore)
        colors[i] = "g"

    return colors


def table(seasons, models, lsFunction):
    head = ["Model"]

    for i in seasons.keys():
        head.append(i)

    rowValues = []
    colorRows = [["lightcyan"]]

    column1 = ["Odds", "Empirical"]
    # column1=[]

    for name in models.keys():
        column1.append(name)

    rowValues.append(column1)

    for season in seasons.keys():
        data = seasons[season]

        column = ["%.4f" % data.getLSOdds(int(data.nMatches() / 2)),
                  "%.4f" % data.getEmpiricalLS(int(data.nMatches() / 2))]
        # column=[]

        for name in models.keys():
            model = Model(data, models[name][0], models[name][1]);
            solver = None;
            value = 0
            value = lsFunction(None, model)
            # value=Presentation.computeMeanLS_SG(None, solver)
            print(value)

            column.append("%.4f" % value)
        rowValues.append(column)
        #colorRows.append(Presentation.getColorGrad(None, column))
    print(rowValues)

    fig = plotly.Figure(data=[plotly.Table(
        header=dict(values=head,
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=rowValues,  # 2nd column
                   line_color='darkslategray',
                   fill_color=colorRows,
                   align='center'))
    ])

    fig.update_layout(width=800, height=800)
    fig.show()

def getNames(infer):

    if isinstance(infer, KalmanFilter):
        if infer.mode=="KF":
            name="KF"
        elif infer.mode=="SKF":
            name="SKF"
        elif infer.mode=="SSKF":
            name="SSKF"
        if infer.model.model=="BradleyTerry":
            return name+"-BT"
        elif infer.model.model=="Thurstone":
            return name+"-T"

    elif isinstance(infer, Elo):
        return "Elo"
    elif isinstance(infer, Glicko):
        return "Glicko"
    elif isinstance(infer, Trueskill):
        return "Trueskill"


def plotArgs(Infers, epsArgs, betArgs, var0Args, K,title, mode="", iter=1):
    """if mode=="NHL":
        data=dataNHL
        K=K_H
    elif mode=="S1":
        data=dataS1
        K=K_S1
    elif mode=="S2":
        data=dataS2
        K=K_S2
    elif mode=="Gauss":
        data=dataGauss"""

    colors=["c", "m", "y", "r", "g", "b"]
    """
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
            EloLS+=getLSOnInfer(elo, P=D.P)/len(data)"""


    def getLS(infers, eps, bet, var0):
        temp=0
        for inf in infers:
            inf.data.resetParam()
            if isinstance(inf, Elo):
                inf.infer(K, bet, var0=var0)
            else:
                inf.infer(eps, bet, var0=var0)
            temp+=getLSOnInfer(inf, P=inf.data.P, start=int(len(inf.data.input)/2))/len(infers)
        return temp

    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    fmt = mticker.FuncFormatter(g)

    head=[]
    rowValues=[]
    colorRows = []
    head.append("eps")
    epsValues = [fmt(e) for e in epsArgs[0]]
    rowValues.append(epsValues)
    colorRows.append(getColorGrad(epsValues, mode="white"))

    for i, infers in enumerate(Infers):

        epsList=epsArgs[i]
        betList=betArgs[i]
        var0List=var0Args[i]

        for var in var0List:
            for beta in betList:
                name = getNames(infers[0])
                string=name +" V0:"+str("%.1f"%var)
                #print(string)
                head.append(string)
                valLS=[]
                for eps in epsList:
                    valLS.append("%.5f" % getLS(infers, eps, beta, var))
                    print(len(valLS))
                rowValues.append(valLS)
                print(valLS)
                colorRows.append(getColorGrad(valLS))


    """    fig = plotly.Figure(data=[plotly.Table(
        header=dict(values=head,
                    line_color='darkslategray',
                    fill_color='lightskyblue',
                    align='left'),
        cells=dict(values=rowValues,  # 2nd column
                    line_color='darkslategray',
                    fill_color=colorRows,
                    align='center'))
        ])

    fig.update_layout(width=1200, height=800)
    fig.write_image(name+".png")"""

    plt.figure(figsize=(15, 10))
    t=plt.table(np.array(rowValues).transpose(),cellColours=np.array(colorRows).transpose(), loc="center", colLabels=head)
    plt.axis("off")
    t.auto_set_font_size(False)
    t.set_fontsize(15)
    t.scale(1.2, 2.2)
    #plt.show()
    plt.savefig(title+".png")
    plt.clf()


        #plt.plot(epsArgs, np.ones(len(epsArgs))*EloLS, "k", label="Elo")

    """def LSon5(params):
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
    """
    """    for beta in betArgs:
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
"""


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
        epsilon = 5e-5
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
            yELO[i]+=(getLSOnInfer(ELOs[j], start=x[i], end=x[i + 1]))/len(data)

    ax.plot(x[1:], yKF, color=colors.pop(), label="KF")
    ax.plot(x[1:], ySKF, color=colors.pop(), label="SKF")
    #ax.plot(x[1:], yELO, color=colors.pop(), label="Elo")

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
            yELO[i]+=(getLSOnInfer(ELOs[j], start=x[i], end=x[i + 1]))/len(data)

    ax.plot(x[1:], yKF, color=colors.pop(), label="KF")
    ax.plot(x[1:], ySKF, color=colors.pop(), label="SKF")
    ax.plot(x[1:], yELO, color=colors.pop(), label="Elo")

    ax.set(ylabel="LS")
    ax.set_title("Mean LS over time intervals")
    ax.legend(loc="upper right")

