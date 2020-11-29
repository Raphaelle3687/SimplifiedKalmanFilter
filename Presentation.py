import numpy as np
import matplotlib.pyplot as plt

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