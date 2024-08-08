import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg

from drone_mpc.common import *
from drone_mpc.sys_dynamics import SysDyn, Predictor

class AcadosCustomOcp:

    def __init__(self):
        self.nx = 0
        self.nu = 0
        self.ny = 0
        self.ns = 0

        self.ocp = None,
        self.solver = None,
        self.integrator = None
        self.sysModel = None

        self.zeta_0 = None
        self.zeta_N = None
        self.u_N = None

        self.ocp = self.setup_acados_ocp()

    def setup_acados_ocp(self):
        '''Formulate Acados OCP'''

        # create casadi symbolic expressions
        sysModel = SysDyn()
        self.sysModel = sysModel

        zeta_f, dyn_f, u, proj_constr, dyn_fn  = sysModel.SetupOde()
        self.zeta_0 = np.copy(init_zeta)

        # sysModel.setupDistNlp()

        # create Acados model
        ocp = AcadosOcp()
        model_ac = AcadosModel()
        model_ac.f_expl_expr = dyn_f
        model_ac.x = zeta_f
        model_ac.u = u
        model_ac.name = "drone_FrenSer"
        ocp.model = model_ac

        # set dimensions
        ocp.dims.N = N
        self.nx = model_ac.x.size()[0]
        self.nu = model_ac.u.size()[0]
        ny = self.nx + self.nu

        self.zeta_N = ca.repmat(np.reshape(self.zeta_0, (self.nx,1)), 1, N+1)
        self.u_N = ca.repmat(U_REF, 1, N)

        # continuity constraints
        ocp.constraints.x0  = self.zeta_0

        # formulate cost function
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = ca.vertcat(model_ac.x, model_ac.u)
        ocp.cost.yref = np.array([ 0.2, 0, 0,  # pf_ref
                                   1, 0, 0, 0,  # q_ref
                                   0, 0, 0,    # pfDot_ref
                                   0, 0, 0,    # omg_ref
                                   0, 0, 0,    # v_ref
                                  U_HOV, U_HOV, U_HOV, U_HOV,  #ohm_ref
                                  0, 0, 0, 0]) # ohmDot_ref
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr_e = model_ac.x
        ocp.cost.yref_e = np.array([ 0.2, 0, 0,
                                     1, 0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0,
                                     0, 0, 0,
                                     U_HOV, U_HOV, U_HOV, U_HOV])
        ocp.cost.W_e = Qn

        # Formulate inquality constraints

        # Constrain AGV dynamics : acceleration, angular velocity (convex ?, Non-linear)
        dyn_constr_eqn = []
        # dyn_constr_eqn = ca.vertcat(dyn_constr_eqn , proj_constr)
        # dyn_constr_len = dyn_constr_eqn.shape[0]

        ineq_constr_eqn = []
        ineq_constr_eqn = ca.vertcat(ineq_constr_eqn, dyn_constr_eqn)

        model_ac.con_h_expr = ineq_constr_eqn
        model_ac.con_h_expr_e = ineq_constr_eqn

        # # Inequality bounds
        # nh = model_ac.con_h_expr.shape[0]

        # # Constrain controls
        # lbu = [0] * self.nu;      ubu = [0] * self.nu

        # # Control bounds ( Affects horizon quality before switch)
        # lbu[0] = OHM_MIN;         ubu[0] = OHM_MAX
        # lbu[1] = OHM_MIN;         ubu[1] = OHM_MAX

        # ocp.constraints.lbu = np.array(lbu)
        # ocp.constraints.ubu = np.array(ubu)
        # ocp.constraints.idxbu = np.array([0, 1])

        # # Bounds on path constraints (inequality)
        # lh = np.zeros(nh);      uh = np.zeros(nh)
        # # Bounds on AGV dynamics (inequality)
        # lh[:obst_constr_len ] = 0;              uh[:obst_constr_len ] = INF
        # lh[obst_constr_len ] = V_MIN;           uh[obst_constr_len ] = V_MAX
        # lh[obst_constr_len + 1] = OMG_MIN;      uh[obst_constr_len + 1] = OMG_MAX
        # lh[obst_constr_len + 2] = -INF;         uh[obst_constr_len + 2] = 1
        # lh[:] = -INF;             uh[:] = 1

        # ocp.constraints.lh = lh
        # ocp.constraints.uh = uh

        # ocp.constraints.lh_e = lh
        # ocp.constraints.uh_e = uh

        # '''Slack constraints (obstacle, vehice dynamics)'''
        # nsh = nh
        # self.ns = nsh
        # # ocp.constraints.idxsh_0 = np.array(range(nsh))
        # ocp.constraints.idxsh = np.array(range(nsh ))
        # ocp.constraints.idxsh_e = np.array(range(nsh))

        # # Define L1 penalty for lower bound violation
        # # dynamics slacks
        # L1_pen_l = 10 * np.ones((nsh,))
        # ocp.cost.zl = L1_pen_l
        # ocp.cost.zl_e = L1_pen_l

        # # Define L1 penalty for upper bound violation
        # # dynamics  slacks
        # L1_pen_u = 10 * np.ones((nsh,))
        # L1_pen_u[:obst_constr_len] = 0
        # ocp.cost.zu = L1_pen_u
        # ocp.cost.zu_e = L1_pen_u

        # # Define L2 penalty for lower bound violation
        # # dynamics slacks
        # L2_pen_l = 1 * np.ones((nsh,))
        # L2_pen_l[:obst_constr_len] = 1e8
        # # L2_pen_l[:-3] = 1e8
        # ocp.cost.Zl = L2_pen_l
        # ocp.cost.Zl_e = L2_pen_l

        # # Define L2 penalty for lower bound violation
        # # dynamics slacks
        # L2_pen_u = 1 * np.ones((nsh,))
        # ocp.cost.Zu = L2_pen_u
        # ocp.cost.Zu_e = L2_pen_u

        # Configure itegrator and QP solver
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.tf = Tf
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        # ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        # ocp.solver_options.time_steps = time_steps
        # ocp.solver_options.shooting_nodes = shooting_nodes

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"#"PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_HPIPM" #"PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx =  "GAUSS_NEWTON"#"EXACT",
        # ocp.solver_options.cost_discretization ="INTEGRATOR"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # ocp.solver_options.sim_method_newton_iter = 10
        # ocp.solver_options.qp_solver_cond_N = N
        # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        # ocp.solver_options.line_search_use_sufficient_descent = True
        ocp.solver_options.nlp_solver_max_iter = 1000
        ocp.solver_options.qp_solver_iter_max = 1000


        # ocp.solver_options.qp_solver_warm_start = True
        ocp.solver_options.tol = 1e-3
        # ocp.solver_options.qp_solver_cond_N = N
        ocp.qp_solver_tol = 1e-3

        # ocp.solver_options.nlp_solver_tol_comp = 1e-1

        # create solver
        solve_json = "planner_ocp.json"
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file = solve_json)
        self.integrator = AcadosSimSolver(ocp, json_file = solve_json) #TODO

        return True


    def solve_and_sim(self):
        '''Solve the OCP with multiple shooting, and forward simulate with RK4'''


        # print(f"DEBUG ctrl {u_0, status}")
        # Integrate ODE model to get CL estimate (point mass)
        # self.zeta_0 = np.reshape(self.solver.get(1, "x"), (self.nx, 1))
        # u_0 = np.copy(U_REF)
        # print(f"DEBUG ctrl {u_0, self.zeta_0}")
        u_0, status = self.solve_for_x0(x0_bar = self.zeta_0)
        self.zeta_0 = self.integrator.simulate(x=self.zeta_0, u=u_0)
        # print(f"DEBUG ctrl {u_0, self.zeta_0}")
        # Store state (entire horizon) and control (only u0) trajectories
        self.zeta_N = np.reshape(self.solver.get(0, "x"), (self.nx, 1))
        for i in range(1, N +1):
            zeta_i = np.reshape(self.solver.get(i, "x"), (self.nx, 1))
            self.zeta_N = np.concatenate((self.zeta_N, zeta_i), axis = 1)

        self.u_N[:, 0] = u_0
        return status

    # user expected to handle status, instead of raising exception
    def solve_for_x0(self, x0_bar):
        '''Wrapper around `solve()` which sets initial state constraint, solves the OCP, and returns u0.'''

        self.solver.set(0, "lbx", x0_bar)
        self.solver.set(0, "ubx", x0_bar)

        status = self.solver.solve()

        if status == 2:
            print("Warning: acados_ocp_solver reached maximum iterations.")
        elif status != 0:
            pass
            #raise Exception(f'acados acados_ocp_solver returned status {status}')

        u0 = self.solver.get(0, "u")
        return u0, status

    def cost_update_ref(self, zeta_0, u_ref):

        s0 = float(zeta_0[0])

        if s0 >= S_MAX:
            return True

        sref =  s0 + S_REF
        srefDot = S_REF / Tf
        for j in range(N):
            sref_j = s0 + (sref - s0) * j /N
            yref = np.array([sref_j, 0, 0, 0, 0, 0, 0, srefDot, 0, 0, 0, 0, 0, 0, 0, 0, U_HOV, U_HOV, U_HOV, U_HOV, 0, 0, 0, 0])
            self.solver.set(j, "yref", yref)

        yref = np.array([sref_j, 0, 0, 0, 0, 0, 0, srefDot, 0, 0, 0, 0, 0, 0, 0, 0, U_HOV, U_HOV, U_HOV, U_HOV])
        self.solver.set(N, "yref", yref)

    def get_ineq_slack(self):
        Sl = np.zeros((self.ns, 2))
        Sl[:, 0] = self.solver.get(1, "sl")
        Sl[:, 1] = self.solver.get(1, "su")
        Sl = ca.reshape(Sl, (self.ns, 2))
        # print(f"\n DEBUG slack {Sl}")
        return Sl

    def get_cost(self):
        cost = self.solver.get_cost()
        return cost

    def get_residuals(self):
        residuals = self.solver.get_residuals()
        # print(f"DEBUG res {residuals}")
        return residuals

    # def update_parameters(self):

    #     for stage in range(N):
    #         self.solver.set(stage, 'p', obst_constr)

        #print(f"\n DEBUG  obst constr {np.round(obst_constr,2)}")

    # def compute_twist(self, zeta_0):
    #     # cmd_vel_twist = [linear  : x: v_agv = cos(alpha) * v, y: 0, z: 0
    #     #                  angular : x: 0,                    , y: 0, z: theta_dt = sin(alpha) * v /d]
    #     if(self.lifted):
    #         v_agv = ca.cos(zeta_0[7]) * zeta_0[6]
    #         omg_agv = ca.sin(zeta_0[7]) * zeta_0[6] / d
    #     else:
    #         v_agv = ca.cos(zeta_0[4]) * zeta_0[3]
    #         omg_agv = ca.sin(zeta_0[4]) * zeta_0[3] / d

    #     return v_agv, omg_agv