import numpy as np
from time import time
import casadi as ca

from drone_mpc.common import *

def plan_ocp( ocp_wrapper):
    '''Motion control of AGV using OCPs'''

    # dimensions
    nx = ocp_wrapper.nx
    nu = ocp_wrapper.nu
    ns = ocp_wrapper.ns

    # initialize iteration variables
    t0 = 0
    discr_step = np.array([])
    times = np.array([[0]])
    mpc_iter = 0
    cost = 0
    fail = 0

    # Define states, controls, and slacks as coloumn vectors
    zeta_0 = np.copy(ocp_wrapper.zeta_0)
    u_0 = np.copy(U_REF)

    # ns x 2 (lower and upper bound))
    Sl_0 = np.zeros((ns, 2))

    # Convert 1D array (nx,) to 3D (nx, 1, N+1)
    zeta_N = ca.repmat(np.reshape(zeta_0, (nx,1)), 1, N+1)

    # nx x N+1 x mpc_iter (to plot state during entire Horizon)
    state_traj = ca.repmat(zeta_0, 1, N+1)
    # nu x mpc_iter
    control_traj = ca.repmat(u_0, 1, 1)
    # 3 x mpc_iter (iteration time, cost)
    sample_traj = ca.repmat(np.array([0, 0]).T, 1)
    # ns x 2 x mpc_iter
    slack_traj = ca.repmat(Sl_0, 1, 1)
    # 2 x mpc_iter

    # Control loop entry point

    for i in range(Nsim):
        # Store previous iterate data for plots
        discr_step = ca.reshape(np.array([t0, cost]), 2, 1)
        sample_traj = np.concatenate( (sample_traj, discr_step), axis = 1)
        state_traj = np.dstack((state_traj, ocp_wrapper.zeta_N))
        control_traj = np.concatenate((control_traj,
                                        np.reshape(ocp_wrapper.u_N[:, 0], (nu, 1))),
                                        axis = 1)
        # slack_traj = np.dstack((slack_traj, Sl_0))
        t1 = time()

        end = ocp_wrapper.cost_update_ref(zeta_N[:, 0], U_HOV)
        if (end or fail >10) :
            print("MPC exit !")
            break

        # Solve the OCP with updated state and controls
        status =  ocp_wrapper.solve_and_sim()
        if(status !=0):
            fail = fail + 1
            print("Optima not found")

        # cost = ocp_wrapper.get_cost()
        # Sl_0 = ocp_wrapper.get_ineq_slack()
        # ocp_wrapper.get_residuals()

        t0 = round(t0 + T_del, 3)
        t2 = time()
        ocp_soln_time = t2-t1
        times = np.vstack(( times, ocp_soln_time))
        mpc_iter = mpc_iter + 1

        zeta_N = ocp_wrapper.zeta_N

        print(f'\n Soln. {mpc_iter} s: {(np.round(ocp_wrapper.zeta_N[0, 0],2).T)} at {round(t0,2)} s\t')

    sqp_max_sec = round(np.array(times).max(), 3)
    sqp_avg_sec = round(np.array(times).mean(), 3)
    avg_n = round(np.abs(state_traj[1, 0, :]).mean(), 4)
    avg_b = round(np.abs(state_traj[2, 0, :]).mean(), 4)
    print(f'\nNumber of NLP failures {fail}')
    print(f'Max. solver time\t: {sqp_max_sec * 1000} ms')
    print(f'Avg. solver time\t: {sqp_avg_sec * 1000} ms')
    print(f'Avg. n\t\t: {avg_n} m')
    print(f'Avg. b\t\t: {avg_b} m')
    # print(f'Avg. lateral deviation N\t: {lat_devN} m')


    return sample_traj, state_traj, control_traj, slack_traj