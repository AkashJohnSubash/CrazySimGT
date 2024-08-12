import numpy as np
import casadi as ca
import os
from pathlib import Path
from typing import Union

'''Global variables'''

track="trefoil_track2.txt"

def getTrack():
    track_file = os.path.join(str(Path(__file__).parent), "tracks/", track)
    array=np.loadtxt(track_file, skiprows=1)
    sref = array[1:,0]
    xref = array[1:,1]
    yref = array[1:,2]
    zref = array[1:,3]

    return sref, xref, yref, zref

[s_ref, x_ref, y_ref, z_ref] = getTrack()

length = len(s_ref)
pathlength = s_ref[-1]

# CF2.1 physical parameters
g0  = 9.80665     # [m.s^2] accerelation of gravity
mq  = 31e-3      # [kg] total mass (with Lighthouse deck)
Ix = 1.395e-5   # [kg.m^2] Inertial moment around x-axis
Iy = 1.395e-5   # [kg.m^2] Inertial moment around y-axis
Iz = 2.173e-5   # [kg.m^2] Inertia moment around z-axis
Cd  = 7.9379e-06 # [N/krpm^2] Drag coefficient
Ct  = 3.25e-4    # [N/krpm^2] Thrust coefficient
dq  = 92e-3      # [m] distance between motors' center
l   = dq/2       # [m] distance between motors' center and the axis of rotation

SLEEP_SEC = 0.06
INF = 1e5

# timing parameters
T_del = 0.02               # time between steps in seconds
N = 50                     # number of shooting nodes
Tf = N * T_del * 1

Tsim = 35
Nsim = int(Tsim * N / Tf)


V_MAX = 0.5       ;   V_MIN = -0.5                    #  [m/s]
W_MAX = np.pi/4   ;   W_MIN = -np.pi/4                #  [rad/s]

U_MAX = 22      # [krpm]
U_HOV = int(np.sqrt(.25 * 1e6* mq * g0 /Ct)) /1000    #[krpm]
U_REF = np.array([U_HOV, U_HOV, U_HOV, U_HOV])
print(f"DEBUG hov KRPM {U_REF}")
# State
n_states = 20

init_zeta = np.array([0.05, 0, 0,       # s,  n,  b
                      1, 0, 0, 0,       # qw, qx, qy, qz,
                      .02, 0, 0,        # sdot, ndot, bdot
                      0, 0, 0,          # ohmr,  ohmp,  ohmy
                      0, 0, 0,          # vx, vy, vz
                      U_HOV, U_HOV, U_HOV, U_HOV ])     # ohm1, ohm2, ohm3, ohm4

# init_zeta = np.array([0.05, -1.14574e-20, -4.71437e-08, 1, 0, 0, 0, 0.2, -2.32704e-18, -4.71437e-06, 0, 0, 0, 0, 0, -4.71406e-06])
rob_rad = 0.04                           # radius of the drone sphere

obst_constr = ([-12.5, -0.75, 1, np.pi/2,
                -8, 10, np.pi/4, 0,
                20, 20, 1, 0,
                20, 20, 0.5, 0])

obst_dim = 4
N_obst_max = int(np.shape(obst_constr)[0]/obst_dim)

# Control
n_controls = 4

ROLL_TRIM  = 0
PITCH_TRIM = 0

# Weights & refeerence
V_XYZ_REF = 0.00
S_REF = 0.25
S_MAX = 5.7
                                          # State weights on
Q = np.diag([1, 1e-1, 1e-1,               # frenet position
             1e-5, 1e-5, 1e-5, 1e-5,      # quaternion
             1e-5, 1e-5, 1e-5,            # frenet velocity
             1e-5, 1e-5, 1e-5,            # drone angular velocity
             1e-5, 1e-5, 1e-5,            # cartesian velocity
             1e-8, 1e-8, 1e-8, 1e-8])     # rotor angular velocity

                                          # Terminal state weights on
Qn = np.diag([10, 1e-3, 1e-2,             # frenet position
             1e-5, 1e-5, 1e-5, 1e-5,      # quaternion
             1e-5, 1e-5, 1e-5,            # frenet velocity
             1e-5, 1e-5, 1e-5,            # drone angular velocity
             1e-5, 1e-5, 1e-5,            # cartesian velocity
             1e-8, 1e-8, 1e-8, 1e-8])     # rotor angular velocity

R = np.diag([1e-5, 1e-5, 1e-5, 1e-5])

#  MatPlotLib parameters
Tstart_offset = 0
f_plot = 15
refresh_ms = 10
sphere_scale = 1 #TODO make dependant on map size. (10000/ 20 obst)
z_const = 0.1

''' Helper functions'''

def DM2Arr(dm):
    return np.array(dm.full())

def quat2eul(qoid):
    ''' qoid -> [qw, qx, qy, qz]
        reference NMPC'''

    R11 = 2*(qoid[0]*qoid[0] + qoid[1]*qoid[1]) - 1
    R21 = 2*(qoid[1]*qoid[2] - qoid[0]*qoid[3])
    R31 = 2*(qoid[1]*qoid[3] + qoid[0]*qoid[2])
    R32 = 2*(qoid[2]*qoid[3] - qoid[0]*qoid[1])
    R33 = 2*(qoid[0]*qoid[0] + qoid[3]*qoid[3]) - 1

    # Euler angles in degrees
    phi	  =  ca.atan2(R32, R33) * 180 / np.pi         # roll
    theta = -ca.asin(R31)       * 180 / np.pi         # pitch
    psi	  =  ca.atan2(R21, R11) * 180 / np.pi         # yaw

    return [phi, theta, psi]

def quat2rpy(qoid):
    ''' qoid -> [qw, qx, qy, qz]
        returns euler angles in degrees
        reference math3d.h crazyflie-firmware'''

    r	  =  ca.atan2( 2 * (qoid[0]*qoid[1] + qoid[2]*qoid[3]), 1 - 2 * (qoid[1]**2 + qoid[2]**2 ))
    p     =  ca.asin( 2 *  (qoid[0]*qoid[2] - qoid[1]*qoid[3]))
    y	  =  ca.atan2( 2 * (qoid[0]*qoid[3] + qoid[1]*qoid[2]), 1 - 2 * (qoid[2]**2 + qoid[3]**2 ))

    r_d = r * 180 / np.pi          # roll in degrees
    p_d = p * 180 / np.pi          # pitch in degrees
    y_d = y * 180 / np.pi          # yaw in degrees

    return [r_d, p_d, y_d]

def eul2quat(eul):
    ''' eul ->  [phi, theta, psi] in degrees
        a.k.a roll, pitch, yaw
        reference NMPC'''

    phi = 0.5* eul[0] * np.pi / 180
    th  = 0.5* eul[1] * np.pi / 180
    psi = 0.5* eul[2] * np.pi / 180

    qw =  ca.cos(phi) * ca.cos(th) * ca.cos(psi) + ca.sin(phi) * ca.sin(th) * ca.sin(psi)
    qx = -ca.cos(psi) * ca.cos(th) * ca.cos(phi) + ca.sin(psi) * ca.sin(th) * ca.cos(phi)
    qy = -ca.cos(psi) * ca.sin(th) * ca.cos(phi) - ca.sin(psi) * ca.cos(th) * ca.sin(phi)
    qz = -ca.sin(psi) * ca.cos(th) * ca.cos(phi) + ca.cos(psi) * ca.sin(th) * ca.sin(phi)

    if(qw < 0):
      qw = -qw
      qx = -qx
      qy = -qy
      qz = -qz

    return [qw, qx, qy, qz]

def krpm2pwm(KrpmAvg):
    '''Convert CF propellor angular speed into PWM values'''
    return int(min(max(((KrpmAvg*1000)-4070.3)/0.2685, 0.0), 60000))

def Thrust2pwm(KrpmAvg):
    return int(max(min(24.5307*(7460.8*np.sqrt(KrpmAvg) - 380.8359), 65535),0))

M_SQRT1_2=0.70710678118654752440
def quatDecompress(comp):

    q_4 = np.zeros(4)
    mask = np.uint32(1 << 9) - 1
    i_largest = (comp >> 30)
    sum_squares = float(0)
    for i in range(3, -1, -1):
        if (i != i_largest):
            mag = np.uint32(comp & mask)
            negbit = np.uint32((comp >> 9) & 0x1)
            comp = comp >> 10
            q_4[i] = M_SQRT1_2 * float(mag) / mask
            if negbit == 1:
                q_4[i] = -q_4[i]
            sum_squares += q_4[i] * q_4[i]
    q_4[i_largest] = float(np.sqrt(1 - sum_squares))

    return q_4


def calc_thrust_setpoint(zeta_0):
    # euler in deg from q1, q2, q3, q4
    q1, q2, q3, q4 = zeta_0[3], zeta_0[4], zeta_0[5], zeta_0[6]
    wy = zeta_0[12]
    ohm1, ohm2, ohm3, ohm4 = zeta_0[16], zeta_0[17], zeta_0[18], zeta_0[19]
    eul_deg = quat2rpy([q1, q2, q3, q4])

    roll_x  = eul_deg[0]                                            # Roll
    pitch_y  = eul_deg[1]                                           # Pitch
    thrust_c  = krpm2pwm((ohm1 + ohm2+ ohm3+ ohm4)/4)       # convert average prop RPM to PWM
    #thrust_c  = Thrust2pwm((ohm1 + ohm2+ ohm3+ ohm4)/4)
    roll_c   = roll_x + ROLL_TRIM
    pitch_c  = (pitch_y + PITCH_TRIM)                               # corrected values
    yawrate = wy * 180 /np.pi                                 # r in deg/s

    return roll_c, pitch_c, yawrate, thrust_c

def get_norm_2(diff):

    norm = ca.sqrt(diff.T @ diff)

    return norm

def get_2norm_2(diff):

    norm = (diff.T @ diff)

    return norm

def get_2norm_W(diff, W):

    norm = (diff.T @ W @diff)

    return norm


def get_norm_W(diff, W):

    norm = ca.sqrt(diff.T @ W @diff)

    return norm

def InterpolLuT(s: Union[ca.MX, float]):
    '''Bspline interpolation of curve x, y, zeta based on longitudinal progress (s)
    <-- xref, yref, zref : position reference curve interpol function'''

    x_ref_curve = ca.interpolant("x_ref", "bspline", [s_ref], x_ref)
    y_ref_curve = ca.interpolant("y_ref", "bspline", [s_ref], y_ref)
    z_ref_curve = ca.interpolant("z_ref", "bspline", [s_ref], z_ref)

    return x_ref_curve(s), y_ref_curve(s), z_ref_curve(s)

def projFrenSerretBasis(s: Union[ca.MX, float]):
    '''Project to the Frenet Serret space
    <-- kap_impl : Curvature
    <-- tau_impl : torsion
    <-- dGamma_ds, d2Gamma_ds2, d3Gamma_ds3 : First 3 derivates of the curve w.r.t arc length
    <-- et_MX, en_MX, eb_MX : tangent , normal , binormal unit vectors'''

    InterOpts = {'degree': [5]}
    y_ref_MX = ca.interpolant("y_ref", "bspline", [s_ref], y_ref, InterOpts)
    z_ref_MX = ca.interpolant("z_ref", "bspline", [s_ref], z_ref, InterOpts)
    x_ref_MX = ca.interpolant("x_ref", "bspline", [s_ref], x_ref, InterOpts)


    [d2GammaX_ds2, dGammaX_ds] = ca.hessian(x_ref_MX(s), s)
    [d2GammaY_ds2, dGammaY_ds] = ca.hessian(y_ref_MX(s), s)
    [d2GammaZ_ds2, dGammaZ_ds] = ca.hessian(z_ref_MX(s), s)

    [d4GammaX_ds4, d3GammaX_ds3] = ca.hessian(d2GammaX_ds2, s)
    [d4GammaY_ds4, d3GammaY_ds3] = ca.hessian(d2GammaY_ds2, s)
    [d4GammaZ_ds4, d3GammaZ_ds3] = ca.hessian(d2GammaZ_ds2, s)

    dGamma_ds = ca.vertcat(dGammaX_ds, dGammaY_ds, dGammaZ_ds)
    d2Gamma_ds2 = ca.vertcat(d2GammaX_ds2, d2GammaY_ds2, d2GammaZ_ds2)
    d3Gamma_ds3 = ca.vertcat(d3GammaX_ds3, d3GammaY_ds3, d3GammaZ_ds3)
    d4Gamma_ds4 = ca.vertcat(d4GammaX_ds4, d4GammaY_ds4, d4GammaZ_ds4)

    kap = ca.norm_2(d2Gamma_ds2)
    # kapBar_MX2 = (1/ kap) * d3Gamma_ds3.T @ d2Gamma_ds2
    kapBar_MX = ca.jacobian(kap, s)

    et_MX = dGamma_ds
    en_MX = d2Gamma_ds2/ kap
    eb_MX = ca.cross(et_MX, en_MX)

    etBar_MX = d2Gamma_ds2
    enBar_MX = (1/ kap**3) * (kap**2 * d3Gamma_ds3 - d2Gamma_ds2 @ d3Gamma_ds3.T @ d2Gamma_ds2)
    ebBar_MX = (1/ kap) * ca.cross(dGamma_ds, d3Gamma_ds3)
    # tau_impl = (1/ kap_impl **2) * ca.dot(d2Gamma_ds2, ca.cross(dGamma_ds, d3Gamma_ds3))
    tau_MX =  ca.dot(enBar_MX, ca.cross(et_MX, eb_MX))
    tauBar_MX =  ca.jacobian(tau_MX, s)

    return kap, tau_MX, et_MX, en_MX, eb_MX, kapBar_MX, tauBar_MX, etBar_MX, enBar_MX, ebBar_MX

def evalFrenSerretBasis(s: ca.MX, kap_MX, tau_MX, et_MX, en_MX, eb_MX):
    '''evaluation functions for curve in Frenet Serret space
    <-- kap_fun, tau_fun : Curvature, torsion casADi functions
    <-- et_fun, en_fun, eb_fun : tangent , normal , binormal casADi functions '''

    tau_fun = ca.Function('tau', [s], [tau_MX])
    kap_fun = ca.Function('kap', [s], [kap_MX])
    et_fun = ca.Function('et', [s], [et_MX])
    en_fun = ca.Function('en', [s], [en_MX])
    eb_fun = ca.Function('en', [s], [eb_MX])

    return kap_fun, tau_fun, et_fun, en_fun, eb_fun

def evalFrenSerretDerv(s: ca.MX, kapBar, tauBar):
    '''evaluation functions for curve in Frenet Serret space
    <-- kap_fun, tau_fun : Curvature, torsion casADi functions
    <-- et_fun, en_fun, eb_fun : tangent , normal , binormal casADi functions '''

    tauBar_fun = ca.Function('tau', [s], [kapBar])
    kapBar_fun = ca.Function('kap', [s], [tauBar])

    return kapBar_fun, tauBar_fun

# def evalFrenSerretBasis(s: Union[ca.MX, float]):
#     x_ref_MX = ca.interpolant("x_ref", "bspline", [s_ref], x_ref)
#     y_ref_MX = ca.interpolant("y_ref", "bspline", [s_ref], y_ref)

#     [d2GammaX_ds2, dGammaX_ds] = ca.hessian(x_ref_MX(s), s)
#     [d2GammaY_ds2, dGammaY_ds] = ca.hessian(y_ref_MX(s), s)

#     k_u_MX = ca.norm_2(ca.vertcat(d2GammaX_ds2, d2GammaY_ds2))

#     # unsigned kappa
#     kappa_u_fn = ca.Function('kappa_u', [s], [k_u_MX])

#     et_fun = ca.Function('et', [s], [dGammaX_ds, dGammaY_ds])
#     #en_fun = ca.Function('en', [s], [d2GammaX_ds2/kappa_u_fn, d2GammaY_ds2/kappa_u_fn])

#     return kappa_u_fn, et_fun#, en_fun
