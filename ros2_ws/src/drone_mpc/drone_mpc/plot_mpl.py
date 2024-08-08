import numpy as np
from numpy import pi
from mpl_toolkits.mplot3d import Axes3D
#import mpl_toolkits.mplot3d.art3d as art3d

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
import shutil

from common import *
from sys_dynamics import SysDyn

# plt.rcParams["backend"] = "TkAgg"

text_usetex = True if shutil.which('latex') else False
params = {
        'text.latex.preamble': r"\usepackage{gensymb} \usepackage{amsmath}",
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'text.usetex': text_usetex,
        'font.family': 'serif'
}

mpl.rcParams.update(params)

def plotFreSertPrm():

    sysModel = SysDyn()
    zeta_f_Mx, _, _, _, _ = sysModel.SetupOde()

    # Plotting routine entry point
    list_kap = []
    list_tau = []
    list_kapBar = []
    list_tauBar = []
    kap, tau, et, en, eb, kapBar, tauBar, _, _, _ = projFrenSerretBasis(zeta_f_Mx[0])
    kap_fun, tau_fun, _, _, _ = evalFrenSerretBasis(zeta_f_Mx[0], kap, tau, et, en, eb)
    kapBar_fun, tauBar_fun = evalFrenSerretDerv(zeta_f_Mx[0], kapBar, tauBar)

    # _, tau2_fun, _, _, _ = evalFrenSerretBasis(zeta_f_Mx[0], kap, tau2, et, en, eb)
    #k1_ref_curve = ca.interpolant("k_ref", "bspline", [zeta_f_Mx[0]], [kappa_ref])
    #TODO compute all Mx array to float
    #print("DEBUG eval \t\t kappa_expl \t\t kappa_impl", s_ref.shape[0])
    sample = -1
    for i in range((s_ref.shape[0])):
       #et_fun1 = et_fun(s_ref[i]/2)
       #print("\t", gam4(s_ref[i]))
       list_kap.append(float(kap_fun(s_ref[i])))
       list_tau.append(float(tau_fun(s_ref[i])))
       list_kapBar.append(float(kapBar_fun(s_ref[i])))
       list_tauBar.append(float(tauBar_fun(s_ref[i])))

    # list_tau2.append(float(tau2_fun(s_ref[i])))
    fig1 = plt.figure(figsize=(16,9), num='Parameteric curve 3D')
    param = fig1.add_subplot(2, 3, 1)
    paramDs = fig1.add_subplot(2, 3, 4)
    # tau2 = fig1.add_subplot(3, 3, 7)

    param.stairs(list_tau[:-1], s_ref, baseline=None,label="$\\tau$", color="coral" )
    param.stairs(list_kap[:-1], s_ref, baseline=None,label="$\\kappa$", color="teal")
    param.set_xlim(0, np.amax(s_ref) + 0.2)
    param.set_ylim(   ymin = np.amin(np.ravel(list_tau[:] + list_kap[:]))*1.10 ,
                    ymax = np.amax(np.ravel(list_tau[:] + list_kap[:]))*1.10)

    param.legend(loc='upper right')
    param.grid()

    paramDs.stairs(list_tauBar[:-1], s_ref, baseline=None,label="$\\tau^{\prime}$", color="coral")
    paramDs.stairs(list_kapBar[:-1], s_ref, baseline=None,label="$\\kappa^{\prime}$", color="teal")
    paramDs.set_xlim(0, np.amax(s_ref) + 0.2)
    paramDs.set_ylim(   ymin = np.amin(np.ravel(list_tauBar[:] + list_kapBar[:])),
                    ymax = np.amax(np.ravel(list_tauBar[:] + list_kapBar[:])))

    paramDs.legend(loc='upper right')
    paramDs.grid()

    # plot cartesian 3D view
    ax3d = fig1.add_subplot(2, 3, (2, 6), projection='3d')
    ax3d.azim = -25
    ax3d.elev = 15
    fig1.add_axes(ax3d)

    # Flight cage dimensions
    cage_x = [-1.5, 1.5]
    cage_y = [-2, 1.5]
    cage_z = [0, 2]

    ax3d.set_xlim3d(left = cage_x[0], right = cage_x[1])
    ax3d.set_ylim3d(bottom = cage_y[0], top = cage_y[1])
    ax3d.set_zlim3d(bottom = cage_z[0], top = cage_z[1])

    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')

    ax3d.scatter(x_ref, y_ref, z_ref, c='cornflowerblue', alpha=0.3)

    fig1.tight_layout()
    plt.show()


def plotOptVars(time_stamps, traj_ST0, traj_U0 ):

    '''Plot1 state and control '''
    sysModel = SysDyn()
    zetaMx, _, _, _, _ = sysModel.SetupOde()

    fig1 = plt.figure(figsize=(16,9), num='Direct Elimination state')

    zetaF = fig1.add_subplot(4, 1, 1)
    zetaC = fig1.add_subplot(4, 1, 2)
    zetaS = fig1.add_subplot(4, 1, 3)
    u = fig1.add_subplot(4, 1, 4)

    time_stamps = time_stamps[Tstart_offset:]
    traj_ST0 = traj_ST0[:, Tstart_offset:]
    traj_U0 = traj_U0[:, Tstart_offset:]
    #traj_twist0 = traj_twist0[:, Tstart_offset:]
    dim_st = np.shape(traj_ST0)
    zetaC_hat = np.zeros((3, dim_st[1]))

    # Project frenet state to cartesian
    #for k in range(N+1):
    s_i = traj_ST0[0, :]
    n_i = traj_ST0[1, :]
    beta_i = traj_ST0[2, :]

    x_i, y_i, z_i = sysModel.Fren2CartT(zetaMx, s_i, n_i, beta_i)
    zetaC_hat[0, : ] = np.ravel(x_i)
    zetaC_hat[1, : ] = np.ravel(y_i)
    zetaC_hat[2, : ] = np.ravel(z_i)

    x = np.ravel(zetaC_hat[0, :-1])
    y = np.ravel(zetaC_hat[1, :-1])
    z = np.ravel(zetaC_hat[2, :-1])

    s = np.ravel(traj_ST0[0, :-1])
    n = np.ravel(traj_ST0[1, :-1])
    b = np.ravel(traj_ST0[2, :-1])

    sDot = np.ravel(traj_ST0[7, :-1])
    nDot = np.ravel(traj_ST0[8, :-1])
    bDot = np.ravel(traj_ST0[9, :-1])

    ohm1 = np.ravel(traj_ST0[-1, :-1])
    ohm2 = np.ravel(traj_ST0[-2, :-1])
    ohm3 = np.ravel(traj_ST0[-3, :-1])
    ohm4 = np.ravel(traj_ST0[-4, :-1])

    zetaC.stairs(x, time_stamps[ :], baseline=None,label="$x$ ($\mathrm{m}$)", color="coral" )
    zetaC.stairs(y, time_stamps[ :], baseline=None,label="$y$ ($\mathrm{m}$)", color="teal")
    zetaC.stairs(z, time_stamps[ :], baseline=None,label="$\\varphi$ ($\mathrm{rad}$)", color="#6052a2ff")
    zetaC.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    # zetaC.set_xlabel('time (s)')
    zetaC.set_ylabel("$\\hat{p}^{c}$")
    zetaC.set_ylim( np.amin(np.ravel(zetaC_hat[:, :])) - .2,
                    np.amax(np.ravel(zetaC_hat[:, :])) + .2)
    zetaC.legend(loc='upper right')
    zetaC.grid()

    zetaF.stairs(s, time_stamps[ :], baseline=None,label="$s$ ($m$)", color="coral" )
    zetaF.stairs(n, time_stamps[ :], baseline=None,label="$n$ ($m$)", color="teal")
    zetaF.stairs(b, time_stamps[ :], baseline=None,label="$b$ ($m$)", color="#6052a2ff")
    zetaF.set_ylim( np.amin(np.ravel(traj_ST0[0 : 3, :-2])) - 0.2,
                    np.amax(np.ravel(traj_ST0[0 : 3, :-2])) + 20)
    zetaF.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    # zetaF.set_xlabel('time (s)')
    zetaF.set_ylabel("$p^{f}$")
    zetaF.set_yscale('symlog')
    zetaF.legend(loc='upper right')
    zetaF.grid()

    zetaS.stairs(sDot, time_stamps[ :], baseline=None,label="$\\dot{s}$ ($m s^{-1}$)", color="coral" )
    zetaS.stairs(nDot, time_stamps[ :], baseline=None,label="$\\dot{n}$ ($m s^{-1}$)", color="teal")
    zetaS.stairs(bDot, time_stamps[ :], baseline=None,label="$\\dot{b}$ ($m s^{-1}$)", color="#6052a2ff")
    zetaS.set_ylim( np.amin(np.ravel(traj_ST0[7:10, :-2])) - 0.2,
                    np.amax(np.ravel(traj_ST0[7:10, :-2])) + 0.2)
    zetaS.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    # zetaU.set_xlabel('time (s)')
    zetaS.set_ylabel('$\\dot{p}^{f}$')
    zetaS.legend(loc='upper right')
    zetaS.grid()

    u.stairs(ohm1, time_stamps[ :], baseline=None,label="$\\ohm_{1}$ ($rad s^{-1}$)", color="lightcoral" )
    u.stairs(ohm2, time_stamps[ :], baseline=None,label="$\\ohm_{2}$ ($rad s^{-1}$)", color="plum")
    u.stairs(ohm3, time_stamps[ :], baseline=None,label="$\\ohm_{3}$ ($rad s^{-1}$)", color="darkseagreen" )
    u.stairs(ohm4, time_stamps[ :], baseline=None,label="$\\ohm_{4}$ ($rad s^{-1}$)", color="lightsteelblue")

    u.set_ylim( np.amin(traj_ST0[-4:,:]) - 0.2,
                np.amax(traj_ST0[-4:,:]) + 0.2)
    u.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    u.set_xlabel('time (s)')
    u.set_ylabel('u')
    u.legend(ncol=2, loc='upper right')
    u.grid()

    fig1.tight_layout()
    plt.show()

def plotResiduals(time_stamps, traj_res):
    '''Plot2 residuals '''


    fig2 = plt.figure(figsize=(16,9), num='Direct Elimination Residuals')

    statAx = fig2.add_subplot(4, 1, 1)
    eqAx = fig2.add_subplot(4, 1, 2)
    ineqAx = fig2.add_subplot(4, 1, 3)
    compAx = fig2.add_subplot(4, 1, 4)

    time_stamps = time_stamps[Tstart_offset:]
    traj_res = traj_res[:, Tstart_offset:]

    stat_res = np.ravel(traj_res[0, :-1])
    eq_res = np.ravel(traj_res[1, :-1])
    ineq_res = np.ravel(traj_res[2, :-1])
    comp_res = np.ravel(traj_res[3, :-1])
    comp_res = np.ravel(traj_res[3, :-1])

    statAx.stairs(stat_res, time_stamps[ :], baseline=None,label="$stat$", color="coral" )
    statAx.set_ylim( np.amin(traj_res[0, :]) - 0.2, np.amax(traj_res[0, :]) + 0.2)
    statAx.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    statAx.legend(loc='upper right')
    statAx.grid()

    eqAx.stairs(eq_res, time_stamps[ :], baseline=None,label="$eq$", color="teal")
    eqAx.set_ylim( np.amin(traj_res[1, :]) - 0.2, np.amax(traj_res[1, :]) + 0.2)
    eqAx.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    eqAx.legend(loc='upper right')
    eqAx.grid()

    ineqAx.stairs(ineq_res, time_stamps[ :], baseline=None,label="$ineq$", color="plum")
    ineqAx.set_ylim( np.amin(traj_res[2, :]) - 0.2, np.amax(traj_res[2, :]) + 0.2)
    ineqAx.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    ineqAx.legend(loc='upper right')
    ineqAx.grid()

    compAx.stairs(comp_res, time_stamps[ :], baseline=None,label="$comp$", color="steelblue")
    compAx.set_ylim( np.amin(traj_res[3, :]) - 0.2, np.amax(traj_res[3, :]) + 0.2)
    compAx.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    compAx.set_xlabel('time (s)')
    compAx.legend(loc='upper right')
    compAx.grid()

    fig2.tight_layout()
    plt.show()

def plotCosts(time_stamps, traj_cost, traj_slack0):

    ''' Plot3 costs, slacks'''

    fig3 = plt.figure(figsize=(16,9), num='Direct Elimination Costs')

    costAx = fig3.add_subplot(3, 1, 1)

    time_stamps = time_stamps[Tstart_offset:]
    traj_cost = traj_cost[:, Tstart_offset:]
    traj_slack0 = traj_slack0[:, :,Tstart_offset:]
    #print("DEBUG slack", np.shape(traj_slack0))
    cost = np.ravel(traj_cost[0, :-1])


    costAx.stairs(cost, time_stamps[ :], baseline=None,label="$cost$", color="steelblue")
    costAx.set_ylim( np.amin(traj_cost[ :]) - 0.2, np.amax(traj_cost[ :]) + 0.2)
    costAx.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    # costAx.set_xlabel('time (s)')
    costAx.legend(loc='upper right')
    costAx.grid()

    # plot lower slacks at 1,1 (top right)
    sl = fig3.add_subplot(3, 1, 2)
    sl.set_ylim(np.amin(np.ravel(traj_slack0[:, 0, :])) - 0.1,
                np.amax(np.ravel(traj_slack0[:, 0, :])) + 0.1)
    sl.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    # sl.set_xlabel('time (s)')
    sl.set_ylabel('lower slacks')

    # # Stack all obstacle axes, but label only one
    sl.stairs(traj_slack0[0, 0, :-1], time_stamps[ :], baseline=None, label="obstacle breach (m)", color="lightcoral")
    for i in range(1, obst_constr_len):
        sl.stairs(traj_slack0[i, 0, :-1], time_stamps[ :], baseline=None,color="lightcoral")
    sl.stairs(traj_slack0[ obst_constr_len, 0, :-1], time_stamps[ :], baseline=None, label="$v_{gg}$ breach ($m s^{-1}$)" ,color="burlywood" )
    sl.stairs(traj_slack0[ obst_constr_len + 1, 0, :-1], time_stamps[ :], baseline=None, label="$\\omega_{gg}$ breach ($rad s^{-1}$)" ,color="steelblue" )
    sl.grid()
    sl.legend(loc='upper right')

    # Plot upper slacks (dynamics only) at (1,3) mid right
    su = fig3.add_subplot(3, 1, 3)
    su.set_ylim(np.amin(np.ravel(traj_slack0[obst_constr_len:, 1, :])) - 0.1,
                np.amax(np.ravel(traj_slack0[obst_constr_len:, 1, :])) + 0.1)
    su.set_xlim(0, np.amax(time_stamps[ :]) + 0.2)
    su.set_xlabel('time (s)')
    su.set_ylabel('upper slacks')

    su.stairs(traj_slack0[ obst_constr_len, 1, : -1], time_stamps[ :], baseline=None, label="$v_{gg}$ breach ($m s^{-1}$)" ,color="burlywood" )
    su.stairs(traj_slack0[ obst_constr_len + 1, 1, : -1], time_stamps[ :], baseline=None, label="$\\omega_{gg}$ breach ($rad s^{-1}$)" ,color="steelblue" )
    #UniSuAx = su.stairs([], [0], baseline=None, label="unique breach ($rad s^{-1}$)" ,color="red" )
    su.grid()
    su.legend(loc='upper right')