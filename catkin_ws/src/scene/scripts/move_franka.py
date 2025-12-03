#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import time
import os
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from visualization_msgs.msg import Marker
import argparse
from tf.transformations import quaternion_from_matrix, quaternion_from_euler, euler_from_quaternion
import yaml
import math
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal, OrientationConstraint, Constraints
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryGoal
from franka_msgs.msg import FrankaState
#from franka_msgs.msg import ErrorRecovery
from controller_manager_msgs.srv import SwitchController
import actionlib

controller_running = True

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

X = config["table_width_m"] / 2
Y = config["table_height_m"] / 2
Z = config["z"]


# Nome del controller da gestire
ARM_CONTROLLER = 'robot_arm_controller'

# Stato interno
in_recovery = False

def stop_controller(controller_name):
    try:
        switch_srv = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        switch_srv(stop_controllers=[controller_name],
                   start_controllers=[],
                   strictness=2)
        rospy.loginfo(f"{controller_name} fermato")
    except rospy.ServiceException as e:
        rospy.logerr(f"Errore stoppando {controller_name}: {e}")

def start_controller(controller_name):
    try:
        switch_srv = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        switch_srv(stop_controllers=[],
                   start_controllers=[controller_name],
                   strictness=2)
        rospy.loginfo(f"{controller_name} riavviato")
    except rospy.ServiceException as e:
        rospy.logerr(f"Errore riavviando {controller_name}: {e}")

def state_callback(msg):
    global in_recovery

    # Reflex Mode
    if msg.robot_mode == FrankaState.ROBOT_MODE_REFLEX:
        if not in_recovery:
            rospy.logwarn("Reflex rilevato → recovery in corso...")
            in_recovery = True

            # Stop controller per evitare nuovi comandi
            stop_controller(ARM_CONTROLLER)

            # Esegui ErrorRecovery
            """try:
                recovery = rospy.ServiceProxy('/franka_control/error_recovery', ErrorRecoveryAction)
                recovery()
                rospy.loginfo("ErrorRecovery chiamata")
            except rospy.ServiceException as e:
                rospy.logerr(f"Errore chiamando ErrorRecovery: {e}")"""
            client = actionlib.SimpleActionClient(
                '/franka_control/error_recovery',
                ErrorRecoveryAction
            )
            client.wait_for_server()
            goal = ErrorRecoveryGoal()  # goal vuoto
            client.send_goal(goal)
            client.wait_for_result()
            rospy.loginfo("ErrorRecovery completata")

    # Quando torna in Idle
    #if msg.robot_mode == FrankaState.ROBOT_MODE_IDLE and in_recovery:
    #print(f"MODE: {msg.robot_mode}")
    if msg.robot_mode == FrankaState.ROBOT_MODE_IDLE and in_recovery:
        print("IDLE!")
        rospy.loginfo("Robot tornato in IDLE → riavvio controller")
        start_controller(ARM_CONTROLLER)
        in_recovery = False

def parse_arguments():
    """Parses command line arguments for the robot's target position."""
    parser = argparse.ArgumentParser(
        description="Move the Franka Panda arm to a specified (x, y, z) position and visualize it as a red sphere in RViz.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-x",
        "--pos_x",
        type=float,
        default=0.2,
        help="Target X coordinate in table frame (float).",
    )
    parser.add_argument(
        "-y",
        "--pos_y",
        type=float,
        default=0.445,
        help="Target Y coordinate in table frame (float).",
    )
    parser.add_argument(
        "-z",
        "--pos_z",
        type=float,
        default=0.2,
        help="Target Z coordinate in table frame (float).",
    )
    parser.add_argument(
        "--sphere_diameter",
        type=float,
        default=0.06,
        help="Sphere diameter for RViz visualization (meters).",
    )
    
    # TODO: to fix 
    parser.add_argument(
        "--no-move",
        action="store_true",
        help="Only visualize the target sphere in RViz without commanding motion.",
    )

    return parser.parse_args()

class PandaArm:
    def __init__(self, frame_id="world"):  
        # Init ROS and MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot_client = actionlib.SimpleActionClient('execute_trajectory', ExecuteTrajectoryAction)
        self.arm = moveit_commander.MoveGroupCommander("arm_group")
        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)
        self.arm.set_pose_reference_frame('world')
        self.frame_id = frame_id

        self.arm.set_goal_position_tolerance(0.01)  # default 0.001
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_goal_joint_tolerance(0.01)

        print(f"Robot reference frame: {self.arm.get_planning_frame()}")


        #rospy.init_node("reflex_handler")
        #rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, state_callback)

        """self.recovery_client = actionlib.SimpleActionClient(
            '/franka_control/error_recovery',
            ErrorRecoveryAction
        )
        rospy.loginfo("Waiting for error_recovery server...")
        self.recovery_client.wait_for_server()
        rospy.loginfo("ErrorRecovery server ready.")"""

    @staticmethod
    def table_to_world_transform(x, y, z):
        """Applies the same transformation used for motion to map table coords to robot coords."""
        rx = Y - y
        ry = X - x
        rz = z + Z # z + <altezza_tavolo>
        return rx, ry, rz
    
    def move_to_point_old(self, vx, vy, vz=0, wait_robot=False):
        print(f"    vx: {vx}, vy:{vy}, vz:{vz} ANGOLO")
        x, y, z = self.table_to_world_transform(vx, vy, vz)
        print(f"    vx: {x}, vy:{y}, vz:{z} WORLD")

        rot = [
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]

        target_pose = Pose()

        # Imposizione della posizione finale dell'end-effector
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # Imposizione vincolo di rotazione del frame di end-effector per posizione finale
        q = quaternion_from_matrix(rot)
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        self.arm.set_start_state_to_current_state()
        #print(time.perf_counter())
        self.arm.set_pose_target(target_pose)

        #rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, state_callback)

        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        #print(time.perf_counter())
        print(f"     Movimento result: {success}")
        
        """if not success:
            rospy.logwarn("    Move failed! Trying error recovery...")
            self.arm.do_error_recovery()
            
            # Dopo la recovery, puoi riprovare a muovere il robot
            rospy.sleep(0.5)  # piccolo delay per sicurezza
            success = self.arm.move_to_point(vx, vy, vz)"""
        
        return success
    
    def move_to_point(self, vx, vy, vz=0.1, wait_robot=False):
        print(f"    vx: {vx}, vy:{vy}, vz:{vz} ANGOLO")
        x, y, z = self.table_to_world_transform(vx, vy, vz)
        print(f"    vx: {x}, vy:{y}, vz:{z} WORLD")

        """rot = [
            [-0.5, 0, 0.866, 0],
            [0, 1, 0, 0],
            [-0.866, 0, -0.5, 0],
            [0, 0, 0, 1]
        ]"""

        rot = [
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]

        target_pose = Pose()

        # Imposizione della posizione finale dell'end-effector
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # define rotation coonstraint
        base_q = quaternion_from_matrix(rot)

        target_pose.orientation.x = base_q[0]
        target_pose.orientation.y = base_q[1]
        target_pose.orientation.z = base_q[2]
        target_pose.orientation.w = base_q[3]

        #define orientation constraint
        reference_frame = 'world'
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = reference_frame
        orientation_constraint.link_name = self.arm.get_end_effector_link()

        # set rotation tolerances: x and y must stay fixed while robot can rotate mallet around the z axis within the (-90,90) range
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = math.pi/12
        orientation_constraint.absolute_z_axis_tolerance = math.pi/2
        orientation_constraint.weight = 1.0

        constraints = Constraints()
        constraints.orientation_constraints.append(orientation_constraint)
        self.arm.set_path_constraints(constraints)

        self.arm.set_start_state_to_current_state()
        #print(time.perf_counter())
        self.arm.set_pose_target(target_pose)

        #rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, state_callback)
        fraction = 0.0
        while fraction < 1.0:
            #if fraction < 1.0:
            print(f"".center(30, '='))
            #print(self.arm.get_current_pose('mallet_link'))
            
            plan_cartesian, fraction = self.arm.compute_cartesian_path([target_pose], 0.01)
            print(f"fraction: {fraction}")

            robot_goal = ExecuteTrajectoryGoal()
            robot_goal.trajectory = plan_cartesian
            self.robot_client.send_goal(robot_goal)
            #self.arm.go(wait=True)
            success = self.arm.execute(plan_cartesian, wait=True)

        self.arm.stop()
        self.arm.clear_pose_targets()
        #self.arm.stop()
        
        #success = self.arm.go(wait=True)
        #self.arm.stop()
        #self.arm.clear_pose_targets()
        #print(time.perf_counter())
        #print(f"     Movimento result: {success}")
        
        return success
    
class TargetVisualizer:
    def __init__(self, frame_id="world"):
        # Latched publisher so the sphere persists in RViz without requiring continuous republishing
        self.pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1, latch=True)
        self.frame_id = frame_id

    def publish_sphere(self, vx, vy, vz, diameter=0.06, rgba=(1.0, 0.0, 0.0, 0.9), ns="target", mid=0, frame_id='world'):

        x, y, z = PandaArm.table_to_world_transform(vx, vy, vz)

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD

        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0

        # Scale is the full length along each axis. For a sphere, set x=y=z to the diameter
        m.scale.x = diameter
        m.scale.y = diameter
        m.scale.z = diameter

        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        # Lifetime 0 -> persists until deleted/overwritten
        m.lifetime = rospy.Duration(5.0)

        self.pub.publish(m)


if __name__ == "__main__":

    args = parse_arguments()

    try:
        # Prevent rospy from parsing our CLI arguments
        rospy.init_node("panda_move", anonymous=True, argv=[])
    except rospy.ROSInitException:
        print("ROS Initialization failed.")
        sys.exit(1)

    # Esempio: chiama la funzione con valori di test
    robot = PandaArm()
    visualizer = TargetVisualizer()

    # NEW
    #robot_state = robot.get_current_state()
    #move_group.set_start_state_to_current_state()

    #vx, vy, vz = 0, 0.65, 1.1   # rispetto al WORLD
    #vx, vy, vz = 1.57, 0.425, 0  # rispetto all'angolo
    vx, vy, vz = args.pos_x, args.pos_y, args.pos_z
    #pvx, pvy, pvz = robot.table_to_world_transform(vx, vy, vz)
    visualizer.publish_sphere(vx, vy, vz, diameter=args.sphere_diameter)

    # Imposta il vincolo di orientazione: x_ee allineato con -y_world
    success = robot.move_to_point(vx, vy, vz)

    moveit_commander.roscpp_shutdown()
