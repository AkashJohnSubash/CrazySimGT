import casadi as ca
from drone_mpc.common import *

'''Global Symbolic variables'''
# State variables

# Position (inertial frame, meter)
x = ca.MX.sym('x')
y = ca.MX.sym('y')
z = ca.MX.sym('z')

# # Quarternion heading (body frame )
q1 = ca.MX.sym('q1')
q2 = ca.MX.sym('q2')
q3 = ca.MX.sym('q3')
q4 = ca.MX.sym('q4')
# phi = ca.MX.sym('phi') #Roll
# tht = ca.MX.sym('theta')
# psi = ca.MX.sym('psi')

# Transaltional velocities (inertial frame, m/s)
vx = ca.MX.sym('vx')
vy = ca.MX.sym('vy')
vz = ca.MX.sym('vz')
v_c = ca.vertcat(vx, vy, vz)

# Angular velocities w.r.t phi(roll), theta(pitch), psi(yaw)
# (body frame, m/s)
wr = ca.MX.sym('wr')
wp = ca.MX.sym('wp')
wy = ca.MX.sym('wy')
omg = ca.vertcat(wr, wp, wy)

# Frenet states
# curve displacements in meter
s = ca.MX.sym('s')
n = ca.MX.sym('n')
b = ca.MX.sym('n')

# curve velocities in meter/ second
sDot = ca.MX.sym('sDot')
nDot = ca.MX.sym('nDot')
bDot = ca.MX.sym('bDot')

# Control variable angles (Motor RPM)
ohm1 = ca.MX.sym('ohm1')
ohm2 = ca.MX.sym('ohm2')
ohm3 = ca.MX.sym('ohm3')
ohm4 = ca.MX.sym('ohm4')

# zeta_c = ca.vertcat(x, y, z, q1, q2, q3, q4, vx, vy, vz, wr, wp, wy)
zeta_f = ca.vertcat(s, n, b, q1, q2, q3, q4, sDot, nDot, bDot, wr, wp, wy, vx, vy, vz, ohm1, ohm2, ohm3, ohm4)


alpha1 = ca.MX.sym('alpha1')
alpha2 = ca.MX.sym('alpha2')
alpha3 = ca.MX.sym('alpha3')
alpha4 = ca.MX.sym('alpha4')
u = ca.vertcat( alpha1, alpha2, alpha3, alpha4)


class SysDyn():

    def __init__(self):

        self.n_samples = 0
        self.solver = None

    def SetupOde(self):
        '''ODEs for system dynamic model'''

        # Rate of change of position
        # dx = 2*(vx * ((q1**2 + q2**2)- 0.5)  + vy *(q2*q3 - q1*q4)         + vz *(q1*q3 + q2*q4))
        # dy = 2*(vx * (q1*q4 + q2*q3)       + vy *(q1**2 + q3**2 - 0.5)     + vz *(q3*q4 - q1*q2))
        # dz = 2*(vx * (q2*q4 - q1*q3)       + vy *(q1*q2 + q3*q4)           + vz *((q1**2 + q4**2) - 0.5))

        # xDot = vx
        # yDot = vy
        # zDot = vz

        D = (Cd / mq) *ca.vertcat(vx*2, vy*2, vz**2)
        F = Ct * ca.vertcat(0, 0, ohm1**2  + ohm2**2  + ohm3**2  + ohm4**2 )
        G = ca.vertcat(0, 0, g0)
        J = np.diag([Ix, Iy, Iz])
        M = ca.vertcat(Ct * l * (ohm1**2 + ohm2**2 - ohm3**2 - ohm4**2),
                      Ct * l * (ohm1**2 - ohm2**2 - ohm3**2 + ohm4**2),
                      Cd * (ohm1**2 - ohm2**2 + ohm3**2 - ohm4**2))

        Rq = ca.vertcat(ca.horzcat( 2 * (q1**2 + q2**2) - 1,    -2 * (q1*q4 - q2*q3),       2 * (q1*q3 + q2*q4)),
                        ca.horzcat( 2 * (q1*q4 + q2*q3),         2 * (q1**2 + q3**2) - 1,   2 * (q1*q2 - q3*q4)),
                        ca.horzcat( 2 * (q1*q3 - q2*q4),         2 * (q1*q2 + q3*q4),       2 * (q1**2 + q4**2) - 1))


        omgDot = ca.inv(J) @ (M - ca.cross(omg, J @ omg))

        # # Rate of change of angles (in qauternion)
        q1Dot = (-(q2 * wr) - (q3 * wp) - (q4 * wy))/2
        q2Dot = ( (q1 * wr) - (q4 * wp) + (q3 * wy))/2
        q3Dot = ( (q4 * wr) + (q1 * wp) - (q2 * wy))/2
        q4Dot = (-(q3 * wr) + (q2 * wp) + (q1 * wy))/2
        # dphi =  wr + wp * ca.sin(phi) * ca.tan(tht) + wy * ca.cos(phi) * ca.tan(tht)
        # dtht =  wp * ca.cos(phi) - wy * ca.sin(phi)
        # dpsi =  wp * ca.sin(phi)/ ca.cos(tht) + wy * ca.cos(phi) / ca.cos(tht)

        #  Rate of change of translational velocity (considering coriolis)
        # du =   -vz * wp  + vy * wy  + 2 * g0 * (q1 * q3 + q2 * q4)
        # dv =    vz * wr  - vx * wy  - 2 * g0 * (q1 * q2 + q3*  q4)
        # dw =   -vy * wr  + vx * wp  - 2 * g0 * (q1**2 + q4**2 - 0.5) + (Ct*( ua**2 + ub**2 + uc**2 + ud**2))/mq

        # Rate of change of translational velocity (considering drag)
        # vxDot =    2 * (Ct/mq) *(q1 * q3 + q2 * q4) *(ua**2  + ub**2  + uc**2  + ud**2 ) #- (Cd/mq) * vx
        # vyDot =   -2 * (Ct/mq) *(q1 * q2 - q3 * q4) *(ua**2  + ub**2  + uc**2  + ud**2 ) #- (Cd/mq) * vy
        # vzDot =   -g0 + 2 * Ct/mq *(q1**2 + q4**2 -0.5) * (ua**2  + ub**2  + uc**2  + ud**2 ) #- (Cd/mq) * vz

        vDot_c = -G + (1/ mq) * Rq @ F - D

        # du =    (Ct/mq) *( ca.cos(phi) * ca.sin(tht) * ca.cos(psi) + ca.sin(phi) * ca.sin(psi)) *(ua**2  + ub**2  + uc**2  + ud**2 ) - (Cd/mq) * vx
        # dv =   (Ct/mq) *( ca.cos(phi) * ca.sin(tht) * ca.sin(psi) - ca.sin(phi) * ca.cos(psi)) *(ua**2  + ub**2  + uc**2  + ud**2 ) - (Cd/mq) * vy
        # dw =   -g0 + (Ct/mq) * ca.cos(phi) * ca.cos(tht) * (ua**2  + ub**2  + uc**2  + ud**2 ) - (Cd/mq) * vz

        # Rate of change of angular velocity
        # wrDot = -(Ct*l*(ua**2 + ub**2 - uc**2 - ud**2) - Iy * wp * wy + Iz * wp * wy) / Ix
        # wpDot = -(Ct*l*(ua**2 - ub**2 - uc**2 + ud**2) + Ix * wr * wy - Iz * wr * wy) / Iy
        # wyDot = -(Cd*  (ua**2 - ub**2 + uc**2 - ud**2) - Ix * wr * wp + Iy * wr * wp) / Iz

        ohm1Dot = alpha1
        ohm2Dot = alpha2
        ohm3Dot = alpha3
        ohm4Dot = alpha4
        # Frenet Serret Dynamics
        kap, tau, et, en, eb, kapBar, tauBar, etBar, enBar, ebBar = projFrenSerretBasis(zeta_f[0])

        sDot = (et.T @ v_c) / (1- kap * n)
        nDot = en.T @ v_c + tau *  sDot @ b
        bDot = eb.T @ v_c - tau * sDot @ n

        kapDot = kapBar * sDot
        tauDot = tauBar * sDot
        etDot = etBar * sDot
        enDot = enBar * sDot
        ebDot = ebBar * sDot

        s2Dot = (et.T @ vDot_c + etDot.T @ v_c)/ (1 - kap * n) + et.T @ v_c * ((n * kapDot + nDot * kap)/ (1 - kap * n)**2 )
        n2Dot = en.T @ vDot_c + enDot.T @ v_c + tauDot * sDot * b + tau * s2Dot * b + tau * sDot * bDot
        b2Dot = eb.T @ vDot_c + ebDot.T @ v_c - tauDot * sDot * n - tau * s2Dot * n - tau * sDot * nDot

        dyn_f = ca.vertcat(sDot, nDot, bDot,
                           q1Dot, q2Dot, q3Dot, q4Dot,
                           s2Dot, n2Dot, b2Dot,
                           omgDot[0], omgDot[1], omgDot[2],
                           vDot_c[0], vDot_c[1] , vDot_c[2],
                           ohm1Dot, ohm2Dot, ohm3Dot, ohm4Dot)

        proj_constr = kap * n
        dyn_fun = ca.Function('f', [zeta_f, u], [dyn_f])

        return zeta_f, dyn_f, u, proj_constr, dyn_fun

    def Fren2CartT(self, zetaMX, s_list, n_list, b_list):
        ''' Frenet to Cartesian transform
        --> s : lateral deviation from reference curve
        --> n : lateral deviation from reference curve
        --> b : vertiacal deviation from reference curve
        --> et : unit tangent vector (3x1)
        --> en : unit normal vector (3x1)
        --> eb : unit binormal vector (3x1)
        <-- x : position (x) projection w.r.t reference curve
        <-- y : position (y) projection w.r.t reference curve
        <-- phi : heading (phi) projection w.r.t reference curve '''
        # TODO fix inverse TF
        len = s_list.shape[0]

        gamma_x, gamma_y, gamma_z  = InterpolLuT(s_list)

        kap_MX, tau_MX, et_MX, en_MX, eb_MX, _, _, _, _, _ = projFrenSerretBasis(zetaMX[0])
        _, _, _, en_fun, eb_fun = evalFrenSerretBasis(zetaMX[0], kap_MX, tau_MX, et_MX, en_MX, eb_MX)
        en_list = []
        eb_list = []
        for i in range(0, len):
            en_list.append(en_fun(s_list[i]))
            eb_list.append(eb_fun(s_list[i]))
        en_arr = np.reshape(en_list, (3, len))
        eb_arr = np.reshape(eb_list, (3, len))
        p_x = gamma_x + en_arr[0, :] * n_list + eb_arr[0, :] * b_list
        p_y = gamma_y + en_arr[1, :] * n_list + eb_arr[1, :] * b_list
        p_z = gamma_z + en_arr[2, :] * n_list + eb_arr[2, :] * b_list

        return p_x, p_y, p_z

class Predictor:

    def rk4_explicit(Dyn_fp, s_MX,  state, ctrl):

        kap, tau, et, en, eb, kapBar, tauBar, _, _, _ = projFrenSerretBasis(s_MX)
        kap_fun, tau_fun, et_fun, en_fun, eb_fun = evalFrenSerretBasis(s_MX, kap, tau, et, en, eb)
        kapBar_fun, tauBar_fun = evalFrenSerretDerv(s_MX, kapBar, tauBar)

        print(f"Rk step0 {init_zeta}")
        print(f"eval FS param {kap_fun(state[0]), tau_fun(state[0]), et_fun(state[0]), en_fun(state[0]), eb_fun(state[0])}")
        k1 = Dyn_fp(state, ctrl)
        k2 = Dyn_fp(state + T_del * k1/2, ctrl)
        k3 = Dyn_fp(state + T_del * k2/2, ctrl)
        k4 = Dyn_fp(state + T_del * k3, ctrl)
        st_next_RK4 = state + (T_del /6) * (k1 + 2 * k2 + 2 * k3 + k4)

        print(f"Rk step1 {st_next_RK4}")
        print(f"eval FS param {kap_fun(st_next_RK4[0]), tau_fun(st_next_RK4[0]), et_fun(st_next_RK4[0]), en_fun(st_next_RK4[0]), eb_fun(st_next_RK4[0])}")
        return st_next_RK4