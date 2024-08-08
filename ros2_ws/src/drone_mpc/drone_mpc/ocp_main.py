import rclpy
import rclpy.node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty
from crazyflie_interfaces.msg import LogDataGeneric, AttitudeSetpoint

import argparse

#from plot_mpl import plotFreSertPrm, plotOptVars
from drone_mpc.acados_ocp import AcadosCustomOcp
from drone_mpc.planner import plan_ocp
#from visualize_mpl import animOptVars

# from cflib.utils import uri_helper
# import cflib.crtp
# from cflib.crazyflie import Crazyflie
# from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
# TODO add ros node
from drone_mpc.common import *
from drone_mpc.measurement import pose_ros_cbk, vel_ros_cbk, tracking_cbk

class LocalPlan(rclpy.node.Node):
    def __init__(self):
        super().__init__('cf_0')
        name = self.get_name()
        prefix = '/' + name
        self.is_connected = True
        self.tracking = True
        self.odometry = Odometry()
        self.ocp = AcadosCustomOcp()
        self.control_queue = None
        self.get_logger().info('Initialization completed...')
        self.create_subscription( PoseStamped,
                                  f'{prefix}/pose',
                                  pose_ros_cbk,
                                  10)

        self.create_subscription( LogDataGeneric,
                                  f'{prefix}/velocity',
                                  vel_ros_cbk,
                                  10)

        self.cmd_pub = self.create_publisher(AttitudeSetpoint,
                                            f'{prefix}/cmd_attitude_setpoint',
                                            10)

        # self.track_sub = self.create_subscription(Empty,
        #                                         f'/all/mpc_trajectory',
        #                                         tracking_cbk,
        #                                         10)

def argparse_init():
   '''Initialization for cmd line args'''

   parser = argparse.ArgumentParser()
   parser.add_argument("-m", "--MatPlotLib", action = 'store_true', help = "Display animations")
   parser.add_argument("-l", "--lab", action='store_true', help="Laborotory flight")
   return parser

def main():
    parser = argparse_init()
    args = parser.parse_args()

    rclpy.init()
    mpc_node = LocalPlan()
    try:
        mpc_node.get_logger().info('local planner node waiting for trajectory message')
        while not(mpc_node.tracking) and rclpy.ok():
            rclpy.spin(mpc_node)
        traj_sample, traj_ST, traj_U, traj_slack = plan_ocp(mpc_node.ocp)

    except KeyboardInterrupt:
        mpc_node.get_logger().info('Keyboard interrupt, shutting down.\n')

    finally:
        mpc_node.get_logger().info('Shutting down mpc node.\n')
        mpc_node.destroy_node()
        rclpy.shutdown()

    #traj_sample, traj_ST, traj_U, traj_slack = plan_ocp(custom_ocp)


    #    if args.lab:
    #        cflib.crtp.init_drivers()
    #        uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E007')
    #        print("Lab flight")
    #        with SyncCrazyflie(uri, cf = Crazyflie(rw_cache='./cache')) as scf:

    # Plot controls and state over simulation period
    # if args.MatPlotLib:
    #     animOptVars(traj_sample, traj_ST, traj_U)
    #     plotOptVars(traj_sample[0, :], traj_ST[:, 0, :], traj_U)
    #     # plotFreSertPrm()

if __name__ == '__main__':
   main()