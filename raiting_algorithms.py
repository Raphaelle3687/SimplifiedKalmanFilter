import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats._continuous_distns import _norm_cdf
from scipy.stats._continuous_distns import _norm_pdf
from scipy.stats import logistic
from scipy.stats import hypsecant


########################################################################
def entropy(p):

    y = -p *np.log(p) - (1-p)*np.log(p)
    return y

########################################################################
def ghL_func(zz, y, PAR, only_L = False ):
    #   only_L=True if we need only the output L (this is done to spare calculation)
    def vw_func(z):
        v = np.zeros_like(z)
        ii = z > -20
        # v[ii] = norm.pdf(z[ii]) / norm.cdf(z[ii])
        v[ii] = _norm_pdf(z[ii]) / _norm_cdf(z[ii])
        ii = z <= -20
        v[ii] = abs(z[ii]) + 1 / abs(z[ii])  # limiting case (avoid division by zero)
        w = v * (z + v)
        return v, w

    model = PAR["rating_model"]

    if isinstance(zz, float):
        zz = np.array([zz])

    #if y not in {0, 1}:
    #    print("Problem here: y is not in the set {0,1}")

    g = 0
    h = 0
    L = 0
    if model == "Thurston":
        sign_y = 2 * y - 1
        # L = norm.cdf(zz * sign_y)
        L = _norm_cdf(zz * sign_y)
        if not only_L:
            (v, w) = vw_func(zz * sign_y)
            g = v * sign_y
            h = w
    elif model == "Bradley-Terry":
        a = PAR["a"] # np.log(10)
        pow_zn = 10 ** (-zz)
        FL = 1 / (1 + pow_zn)
        if y == 0:
            L = 1 - FL
        elif y == 1:
            L = FL
        else:
            raise  Exception("y must be = 1 or =0")
        if not only_L:
            g = a * (y - FL)
            # h = np.zeros_like(zz)
            # ii = np.abs(zz) > 2.e+1
            # h[ii] = (a ** 2) * (10 ** (-np.abs(zz[ii])))  # formula for large abs(zz)
            # ii = np.abs(zz) <= 2.e+1
            # h[ii] = (a ** 2) / ((10 ** (zz[ii] / 2) + 10 ** (-zz[ii] / 2)) ** 2)
            h = (a ** 2) * FL * (1-FL)
    elif model == "Gauss":
        var = PAR["std.dev"]**2
        g = (y - zz) / var
        h = np.ones_like(zz) / var
        L = 0
    else:
        print("wrong model")

    return (g, h, L)


########################################################################
def p_hat_GH(m, omega, y, PAR):
    if PAR["rating_model"] == "Thurston":
        #   use analytical integration formula
        sign_y = 2 * y - 1
        # p = norm.cdf(sign_y * m / np.sqrt(1 + omega))
        p = _norm_cdf(sign_y * m / np.sqrt(1 + omega))
    else:
        if omega == 0:
            (g, h, L) = ghL_func(m, y, PAR, only_L=True)
            p = L
        else:   # Outcome probability calculated using Gauss-Hermite quadrature
            K = 10
            (x, w) = np.polynomial.hermite.hermgauss(K)  # Gauss-Hermite quadrature points
            (g, h, L) = ghL_func(x * np.sqrt(2 * omega) + m, y, PAR, only_L=True)
            p = np.dot(L, w) / np.sqrt(np.pi)

    return p


########################################################################
def forecast_metric(zz, y, omega, PAR, p1_true):
    if PAR["rating_model"] == "Gauss":
        LS = (zz - y) ** 2
    else:
        p_th = 1.e-10
        if PAR["metric_type"] == "exact":
            p1_hat = p_hat_GH(zz, omega, 1, PAR)  # whole distribution
            p0_hat = 1 - p1_hat
            if p0_hat < p_th:
                p0_hat = p_th
            if p1_hat < p_th:
                p1_hat = p_th
            LS = - (1 - p1_true) * np.log(p0_hat) - p1_true * np.log(p1_hat)
        if PAR["metric_type"] == "DKL":
            p1_hat = p_hat_GH(zz, 0, 1, PAR)  # use point estimate of skills: omega <- 0
            p0_hat = 1 - p1_hat
            if p0_hat < p_th:
                p0_hat = p_th
            if p1_hat < p_th:
                p1_hat = p_th

            if p1_true < p_th or (p1_true > 1-p_th):
                LS = 0
            else:
                LS = (1 - p1_true) * np.log(1 - p1_true) + p1_true * np.log(p1_true)

            LS += - (1 - p1_true) * np.log(p0_hat) - p1_true * np.log(p1_hat)
        elif PAR["metric_type"] == "empirical":
            p_hat = p_hat_GH(zz, omega, y, PAR)
            if p_hat < p_th:
                p_hat = p_th
            LS = -np.log(p_hat)  # log-score
        elif PAR["metric_type"] == "empirical_0":
            p_hat = p_hat_GH(zz, 0, y, PAR)
            if p_hat < p_th:
                p_hat = p_th
            LS = -np.log(p_hat)  # log-score
        elif PAR["metric_type"] == "empirical_avg":
            # score averaged using Gauss-Hermite quadrature
            K = 10
            (x, w) = np.polynomial.hermite.hermgauss(K)  # Gauss-Hermite quadrature points
            (g, h, L) = ghL_func(x * np.sqrt(2 * omega) + zz, y, PAR, only_L=True)
            ll = -np.log(L)
            LS = np.dot(ll, w) / np.sqrt(np.pi)
        else:
            raise Exception("wrong metric calculation type")

    return LS


####################################################################################################
####################################################################################################
def Kalman(results, PAR):
    # results: Pandas frame with columns {"home_player","away_player","game_result","time_stamp"}
    # PAR: dictionary

    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]
    KFtype = PAR["rating_algorithm"]
    it_nr = PAR["it"]
    PAR_gen = PAR["PAR_gen"]
    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    PAR["a"] = np.log(10)

    N = len(results)  # number of games

    unique_players = set(results["home_player"].unique())
    unique_players.update(results["away_player"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    F = 1  # number of players in each group (home or away)
    skills = np.zeros((N, M))  # all skills through time
    theta = np.zeros(M)  # holder of skills
    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)
    if KFtype == "KF":
        V_t = np.identity(M) * v0  # covariance matrix
    elif KFtype == "vSKF":
        V_t = np.ones(M) * v0  # variance vector
    elif KFtype == "sSKF":
        V_t = v0  # scalar variance
    elif KFtype == "fSKF" or KFtype == "SG":
        None
    else:
        raise Exception('Kalman type not defined')

    ####   main loop
    for n in range(N):
        i_t = player2index[results["home_player"][n]]
        j_t = player2index[results["away_player"][n]]
        #   z_t = theta[i_t] - theta[j_t]
        #   z_t += home
        y_t = results["game_result"][n]
        prob1_real = results["real_proba"][n]

        if n > 0:
            delta_t = results["time_stamp"][n] - results["time_stamp"][n - 1]
        else:
            delta_t = 0
        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            if KFtype == "KF":
                V_tmp = V_t[n_switch:, n_switch:]
                V_t   = np.identity(M ) * v0  # matrix
                V_t[n_switch:, n_switch:] = V_tmp
            elif KFtype == "vSKF":
                V_t[:n_switch] = v0
            elif KFtype == "sSKF":
                V_t = (v0-V_t)* n_switch/M + V_t

        #   find the posterior mode
        z_old_t = theta[i_t] - theta[j_t]
        z0 = beta_t*z_old_t

        for it in range(it_nr):  #  this is the possibility for the iterative improvement (not used in the paper)
            (g_t, h_t, L_t) = ghL_func(z0 / scale, y_t, PAR)
            if KFtype == "SG":
                h_t = 0

            # prepare update
            if it == 0:
                if KFtype == "KF":
                    V_bar = (beta_t ** 2) * V_t + epsilon_t * np.identity(M)  # matrix
                    omega_t = V_bar[i_t, i_t] + V_bar[j_t, j_t] - 2 * V_bar[i_t, j_t]
                    vv = V_bar[:, i_t] - V_bar[:, j_t]
                elif KFtype == "vSKF":
                    V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector
                    omega_t = V_bar[i_t] + V_bar[j_t]
                    vv = np.zeros(M)
                    vv[i_t] = V_bar[i_t]
                    vv[j_t] = -V_bar[j_t]
                elif KFtype == "sSKF":
                    V_bar = (beta_t ** 2) * V_t + epsilon_t  # scalar
                    omega_t = 2 * F * V_bar
                    vv = np.zeros(M)
                    vv[i_t] = V_bar
                    vv[j_t] = -V_bar
                elif KFtype == "fSKF":
                    omega_t = 2 * F * v0
                    vv = np.zeros(M)
                    vv[i_t] = v0
                    vv[j_t] = -v0
                elif KFtype == "SG":
                    omega_t = 0
                    vv = np.zeros(M)
                    vv[i_t] = v0
                    vv[j_t] = -v0

            # update skills
            corr_factor = (z0 - beta_t*z_old_t)
            theta = beta_t * theta + vv * (scale * g_t + h_t * corr_factor) / (scale ** 2 + h_t * omega_t)
            z0 = theta[i_t] - theta[j_t]

        #   prediction
        LS_t = forecast_metric(beta* z_old_t / scale, y_t, omega_t / (scale ** 2), PAR, prob1_real)
        LS[n] = LS_t
        if is_data:
            MSE_t = np.linalg.norm(theta - theta_org[:, results["time_stamp"][n]])**2
            MSE[n] = MSE_t

        #   update variance
        if KFtype == "KF":
            V_t = V_bar - np.outer(vv, vv * (h_t / (scale ** 2 + h_t * omega_t)))
        elif KFtype == "vSKF":
            V_t = V_bar * (1 - np.abs(vv) * (h_t / (scale ** 2 + h_t * omega_t)))
        elif KFtype == "sSKF":
            V_t = V_bar * (1 - omega_t / M * (h_t / (scale ** 2 + h_t * omega_t)))

        skills[n, :] = theta
        if KFtype in {"KF", "vSKF", "sSKF"}:
            V.append(V_t.copy())

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return (skills_frame, LS, V, MSE)


####################################################################################################
####################################################################################################
def TrueSkill(results, PAR):
    # results: Pandas frame with columns {"home_player","away_player","game_result","time_stamp"}
    # PAR: dictionary

    #def r_func(v):
    #    return 1/np.sqrt(1+v*aa/(scale ** 2) )

    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    if beta != 1:
        raise Exception("shoudn't beta equal one?")
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]
    KFtype = PAR["rating_algorithm"]
    if KFtype != "TrueSkill":
        raise Exception("rating algorithm should say: TrueSkill")
    if PAR["rating_model"] != "Thurston":
        raise Exception("TrueSkill only works for Thurston model (just check: not that it is really used somewhere)")
    PAR_gen = PAR["PAR_gen"]
    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    aa = 3 * (np.log(10)/np.pi) ** 2 # to be used in r_func()
    PAR["a"] = np.log(10)

    N = len(results)  # number of games

    unique_players = set(results["home_player"].unique())
    unique_players.update(results["away_player"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    skills = np.zeros((N, M))  # all skills through time
    theta = np.zeros(M)  # holder of skills
    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)

    # elif KFtype == "vSKF":
    V_t = np.ones(M) * v0  # variance vector

    ####   main loop
    for n in range(N):
        i_t = player2index[results["home_player"][n]]
        j_t = player2index[results["away_player"][n]]
        #   z_t = theta[i_t] - theta[j_t]
        #   z_t += home
        y_t = results["game_result"][n]
        prob1_real = results["real_proba"][n]

        if n > 0:
            delta_t = results["time_stamp"][n] - results["time_stamp"][n - 1]
        else:
            delta_t = 0
        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            V_t[:n_switch] = v0

        #   find the posterior mode
        z_old_t = theta[i_t] - theta[j_t]

        # prepare update
        V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector
        omega_t = V_bar[i_t] + V_bar[j_t]
        vv = np.zeros(M)
        vv[i_t] = V_bar[i_t]
        vv[j_t] = -V_bar[j_t]

        scale_tilde = scale * np.sqrt(1+omega_t/(scale ** 2))

        (g_t, h_t, L_t) = ghL_func(beta_t * z_old_t / scale_tilde, y_t, PAR)

        # update skills
        theta = beta_t * theta + vv * g_t / scale_tilde
        #   update variance
        V_t = V_bar * (1 - np.abs(vv) * (h_t / (scale ** 2 + omega_t)))

        #   prediction
        LS_t = forecast_metric(beta* z_old_t / scale, y_t, omega_t / (scale ** 2), PAR, prob1_real)
        LS[n] = LS_t

        skills[n, :] = theta

        V.append(V_t.copy())

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return (skills_frame, LS, V, MSE)
####################################################################################################
####################################################################################################
def Glickog(results, PAR):
    # results: Pandas frame with columns {"home_player","away_player","game_result","time_stamp"}
    # PAR: dictionary

    def r_func(v):
        return np.sqrt(1+v*aa/(scale ** 2) )

    home = PAR["home"]  # HFA
    beta = PAR["beta"]
    if beta != 1:
        raise Exception("shoudn't beta equal one?")
    epsilon = PAR["epsilon"]
    v0 = PAR["v0"]
    scale = PAR["scale"]
    KFtype = PAR["rating_algorithm"]
    if KFtype != "Glicko":
        raise Exception("rating algorithm Should say: Glicko")
    if PAR["rating_model"] != "Bradley-Terry":
        raise Exception("Glicko only works for Bradley-Terry model (not that it realy matters here)")
    PAR_gen = PAR["PAR_gen"]
    is_data = False
    if "data" in PAR.keys():
        is_data = True
        theta_org = PAR["data"]

    aa = 3 * (np.log(10)/np.pi) ** 2 # to be used in r_func()
    PAR["a"] = np.log(10)

    N = len(results)  # number of games

    unique_players = set(results["home_player"].unique())
    unique_players.update(results["away_player"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    if PAR_gen["scenario"] == "switch":
        t_switch = PAR_gen["d_switch"] * int(M/2)

    skills = np.zeros((N, M))  # all skills through time
    theta = np.zeros(M)  # holder of skills
    V = list([])  # covariance matrices/vectors/scalars
    LS = np.zeros(N)
    MSE = np.zeros(N)

    # elif KFtype == "vSKF":
    V_t = np.ones(M) * v0  # variance vector

    ####   main loop
    for n in range(N):
        i_t = player2index[results["home_player"][n]]
        j_t = player2index[results["away_player"][n]]
        #   z_t = theta[i_t] - theta[j_t]
        #   z_t += home
        y_t = results["game_result"][n]
        prob1_real = results["real_proba"][n]

        if n > 0:
            delta_t = results["time_stamp"][n] - results["time_stamp"][n - 1]
        else:
            delta_t = 0
        beta_t = beta ** delta_t  # time-dependent version of beta
        epsilon_t = epsilon * delta_t

        if PAR_gen["scenario"] == "switch" and n == t_switch:  ## special treatement for the switch-time
            n_switch = PAR_gen["n_switch"]      #   number of players switched
            theta[:n_switch] = 0
            V_t[:n_switch] = v0

        #   find the posterior mode
        z_old_t = theta[i_t] - theta[j_t]

        # prepare update
        V_bar = (beta_t ** 2) * V_t + epsilon_t  # vector

        omega_t = V_bar[i_t] + V_bar[j_t]

        scale_i = scale * r_func( V_bar[j_t] )
        (gg_i, hh_i, L_t) = ghL_func(beta_t * z_old_t / scale_i, y_t, PAR)

        scale_j = scale * r_func(V_bar[i_t])
        (gg_j, hh_j, L_t) = ghL_func(beta_t * z_old_t / scale_j, y_t, PAR)

        # update skills
        theta = beta_t * theta
        theta[i_t] += V_bar[i_t] * (scale_i * gg_i) / (scale_i ** 2 + hh_i * V_bar[i_t])
        theta[j_t] -= V_bar[j_t] * (scale_j * gg_j) / (scale_j ** 2 + hh_j * V_bar[j_t])

        #   update variance
        V_t = V_bar
        V_t[i_t] *= (scale_i ** 2 / (scale_i ** 2 + hh_i * V_bar[i_t]))
        V_t[j_t] *= (scale_j ** 2 / (scale_j ** 2 + hh_j * V_bar[j_t]))

        #   prediction
        LS_t = forecast_metric(beta* z_old_t / scale, y_t, omega_t / (scale ** 2), PAR, prob1_real)
        LS[n] = LS_t

        skills[n, :] = theta

        V.append(V_t.copy())

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return (skills_frame, LS, V, MSE)
####################################################################################################
####################################################################################################
def batch_rating(results, PAR):

    v0 = PAR["v0"]
    scale = PAR["scale"]
    step = PAR["step"]

    N = len(results)  # number of games

    unique_players = set(results["home_player"].unique())
    unique_players.update(results["away_player"].unique())
    unique_players = list(unique_players)

    M = len(unique_players)  # number of players
    player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    F = 1  # number of players in each group (home or away)
    theta = np.zeros(M)  # holder of skills

    gg = np.zeros(M)
    grad_th = 1.e-4


    ii = np.array([player2index[results["home_player"][n]] for n in range(N)])
    jj = np.array([player2index[results["away_player"][n]] for n in range(N)])
    yy = np.array(results["game_result"])
    #HH = [ii == m for m in range(M)]    #   indices to the home games of the players 0,..., M-1
    #AA = [jj == m for m in range(M)]     #   indices to the away games of the players 0,..., M-1

    while True:
        zz = theta[ii] - theta[jj]
        (g, h, L) = ghL_func(zz, yy, PAR)

        grad = np.zeros(M)
        hess = np.zeros((M,M))
        for n in range(N):
            grad[ii[n]] += g[n]
            grad[jj[n]] -= g[n]
            hess[ii[n], ii[n]] += h[n]
            hess[jj[n], jj[n]] += h[n]
            hess[ii[n], jj[n]] -= h[n]
            hess[jj[n], ii[n]] -= h[n]

        grad /= scale
        hess /= (scale**2)
        hess += np.identity(M)/v0

        dd = np.linalg.solve(hess, grad)

        if np.linalg.norm(grad) < grad_th*M:
            break

        theta += step * dd

    skills_dict = {unique_players[i]: theta[i] for i in range(M)}

    return (skills_dict, grad, hess)

####################################################################################################
####################################################################################################
def Elo(results, sigma, K, PAR):
    # results: Pandas frame with columns {"home_player","away_player","game_result","time_stamp"}
    # sigma : scaling
    # K: adaptation step
    # PAR: dictionary {"kappa", "home"}

    def e10(x):
        return 10 ** (x / sigma)

    def F_kappa(x):
        return (e10(x / 2) + 0.5 * kappa) / (e10(-x / 2) + e10(x / 2) + kappa)

    def Proba(x):
        pout = np.zeros()
        return (e10(x / 2) + 0.5 * kappa) / (e10(-x / 2) + e10(x / 2) + kappa)

    kappa = PAR["kappa"]  # draw parameter
    home = PAR["home"]  # HFA

    N = len(results)  # number of games

    unique_players = set(results["home_player"].unique())  # set
    unique_players.update(results["away_player"].unique())
    unique_players = list(unique_players)  # convert to list

    M = len(unique_players)  # number of players
    Player2index = {unique_players[i]: i for i in range(M)}  # dictionary assigning player identifies to indices

    skills = np.zeros((N, M))  # all skills through time
    proba = np.zeros((N, 3))  # all skills through time
    theta = np.zeros(M)  # holder of skills

    ####   main loop
    for n in range(N):
        i_t = Player2index[results["home_player"][n]]
        j_t = Player2index[results["away_player"][n]]
        z_t = theta[i_t] - theta[j_t]
        z_t += home * sigma
        ex_score = F_kappa(z_t)
        y_t = results["game_result"][n]

        # update skills
        theta[i_t] += (K * sigma) * (y_t - ex_score)
        theta[j_t] -= (K * sigma) * (y_t - ex_score)

        skills[n, :] = theta

    skills_frame = pd.DataFrame(skills, columns=unique_players)

    return skills_frame
########################################################################