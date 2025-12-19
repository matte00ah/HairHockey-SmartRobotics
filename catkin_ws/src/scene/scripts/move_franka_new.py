#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import os
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker
import argparse
from tf import TransformListener
from tf.transformations import quaternion_from_matrix, quaternion_from_euler, euler_from_quaternion
import yaml
import math
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal, OrientationConstraint, Constraints
from franka_msgs.msg import ErrorRecoveryAction, ErrorRecoveryGoal
from franka_msgs.msg import FrankaState
import numpy as np
#from franka_msgs.msg import ErrorRecovery
from controller_manager_msgs.srv import SwitchController
import actionlib
import time
from std_srvs.srv import Empty

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
REFLEX_MODE = 4

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
        default=0.8,
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

def quaternion_slerp(q0, q1, t):
    dot = (q0[0]*q1[0]) + (q0[1]*q1[1]) + (q0[2]*q1[2]) + (q0[3]*q1[3])
    if dot < 0.0:
        q1 = [-q for q in q1]
        dot = -dot
        DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # Linear interpolation for very close quaternions
        result = [q0[i] + t(q1[i]-q0[i]) for i in range(4)]
        norm = math.sqrt(sum(x for x in result))
        return [x/norm for x in result]
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return [(s0*q0[i]) + (s1*q1[i]) for i in range(4)]

def _vec_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

class PandaReflexRecovery:
    def __init__(self):
        #rospy.init_node("panda_reflex_recovery", anonymous=True)
        
        # Subscriber al topic dello stato del robot
        self.robot_mode = None
        rospy.Subscriber("/franka_state_controller/franka_states", FrankaState, self.state_callback)

        # Service di recovery
        rospy.wait_for_service('/franka_control/error_recovery')
        self.error_recovery_srv = rospy.ServiceProxy('/franka_control/error_recovery', Empty)
        rospy.loginfo("Classe PandaReflexRecovery pronta")

    def state_callback(self, msg):
        self.robot_mode = msg.robot_mode
        # Se siamo in reflex mode (3), chiamiamo la recovery
        if self.robot_mode == 3:
            self.recover()

    def recover(self):
        try:
            self.error_recovery_srv()
            rospy.loginfo("Error recovery eseguito perché robot in REFLEX mode!")
        except rospy.ServiceException as e:
            rospy.logerr("Errore chiamando error_recovery: %s", e)

    
class FrankaAutoRecovery:
    def __init__(self):
        #rospy.init_node("franka_auto_recovery")
        self.in_recovery = False

        # Abilita subscriber allo stato del robot
        rospy.Subscriber("/franka_state_controller/franka_states",
                         FrankaState, self.state_callback)

        # Service per controllare i controller
        rospy.wait_for_service("/controller_manager/switch_controller")
        self.switch_controller = rospy.ServiceProxy(
            "/controller_manager/switch_controller", SwitchController
        )

        # Client dell’action di error recovery
        self.recovery_client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        self.recovery_client.wait_for_server()

        rospy.loginfo("Franka auto recovery attivo")

    def state_callback(self, state):
        if state.robot_mode == REFLEX_MODE and not self.in_recovery:
            rospy.logwarn("⚠️ Robot in REFLEX MODE! Avvio recovery...")
            self.do_recovery()

    def stop_robot_arm_controller(self):
        rospy.loginfo("🔴 Stop robot_arm_controller...")
        self.switch_controller([], ["robot_arm_controller"], 1, False, False)

    # ---------------------------------------------------------------------- #
    # RIAVVIO CONTROLLER
    # ---------------------------------------------------------------------- #
    def start_robot_arm_controller(self):
        rospy.loginfo("🟢 Start robot_arm_controller...")
        self.switch_controller(["robot_arm_controller"], [], 2, False, False)

    # ---------------------------------------------------------------------- #
    # PROCEDURA DI RECOVERY
    # ---------------------------------------------------------------------- #
    def do_recovery(self):
        self.in_recovery = True

        # Step 1: Stop controller
        #self.stop_robot_arm_controller()

        # Step 2: Avvia error recovery
        goal = ErrorRecoveryGoal()
        rospy.loginfo("🔧 Running /franka_control/error_recovery...")
        self.recovery_client.send_goal(goal)
        self.recovery_client.wait_for_result()

        rospy.loginfo("✔ Robot tornato in Idle")

        rospy.sleep(1)

        # Step 3: Riattiva robot_arm_controller
        self.start_robot_arm_controller()

        rospy.loginfo("🟢 Recovery completata — robot pronto!")
        self.in_recovery = False

class PandaArm:
    def __init__(self, frame_id="world"):  
        # Init ROS and MoveIt
        
        self.pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose', 
            PoseStamped, 
            queue_size=1
        )
        rospy.init_node("panda_move", anonymous=True, argv=[])
        moveit_commander.roscpp_initialize(sys.argv)
        #self.robot_client = actionlib.SimpleActionClient('execute_trajectory', ExecuteTrajectoryAction)
        # self.arm = moveit_commander.MoveGroupCommander("arm_group")
        # self.arm.set_max_velocity_scaling_factor(0.1)
        # self.arm.set_max_acceleration_scaling_factor(0.1)
        # self.arm.set_pose_reference_frame('world')
        # self.frame_id = frame_id

        # self.arm.set_goal_position_tolerance(0.03)  # default 0.001
        # self.arm.set_goal_orientation_tolerance(0.03)
        # self.arm.set_goal_joint_tolerance(0.03)

        #print(f"Robot reference frame: {self.arm.get_planning_frame()}")

        #FrankaAutoRecovery()

        #PandaReflexRecovery()

        rospy.sleep(1.0)

    @staticmethod
    def table_to_world_transform(x, y, z):
        """Applies the same transformation used for motion to map table coords to robot coords."""
        rx = Y - y
        ry = X - x
        rz = z + Z # z + <altezza_tavolo>
        return rx, ry, rz
    
    @staticmethod
    def table_to_robot(x, y, z):
        rx = x + 0.40 
        ry = Y - y
        rz = z + Z # z + <altezza_tavolo>
        return rx, ry, rz

    def compute_target_orientation(self, vx, vy, vz, reference_frame, robot_base_frame):
        try:
            yaw_angle_rad =  math.atan2(vy, vx) - (math.pi/2)
            #print(yaw_angle_rad)
            c = np.cos(yaw_angle_rad)
            s = np.sin(yaw_angle_rad)

            Rz = np.array([
                [c, -s, 0, 0],
                [s,  c, 0, 0],
                [0,  0, 1, 0],
                [0, 0, 0, 1]
            ])
            return Rz
            
        except Exception as e:
            rospy.logerr(f"Errore di trasformazione: {e}")
    
    def move_to_point_old(self, vx, vy, vz=0.1, wait_robot=False):
        print(f"    vx: {vx}, vy:{vy}, vz:{vz} ANGOLO")
        x, y, z = self.table_to_world_transform(vx, vy, vz)
        #print(f"    vx: {x}, vy:{y}, vz:{z} WORLD")

        """rot = [
            [-0.5, 0, 0.866, 0],
            [0, 1, 0, 0],
            [-0.866, 0, -0.5, 0],
            [0, 0, 0, 1]
        ]"""

        rot_z = self.compute_target_orientation(x, y, z, 'world', 'mallet_link')

        rot = np.array([
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        target_orientation = rot @ rot_z
        target_orientation = target_orientation.tolist()

        target_pose = Pose()

        # Imposizione della posizione finale dell'end-effector
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # define rotation coonstraint
        base_q = quaternion_from_matrix(target_orientation)

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
        
        for i in range(2):
            #if fraction < 1.0:
            print(f"".center(30, '='))
            #print(self.arm.get_current_pose('mallet_link'))
            
            plan_cartesian, fraction = self.arm.compute_cartesian_path([target_pose], 0.05)
            print(f"fraction: {fraction}")

            robot_goal = ExecuteTrajectoryGoal()
            robot_goal.trajectory = plan_cartesian
            #self.robot_client.send_goal(robot_goal)
            #self.arm.go(wait=True)       

            success = self.arm.execute(plan_cartesian, wait=True)

        #self.arm.stop()
        self.arm.clear_pose_targets()
        #self.arm.stop()
        
        #success = self.arm.go(wait=True)
        #self.arm.stop()
        #self.arm.clear_pose_targets()
        #print(time.perf_counter())
        #print(f"     Movimento result: {success}")
        
        return success

    def move_to_point_new(self, vx, vy, vz=0.1, wait_robot=False, linear_speed=0.05, angular_speed=0.8, rate_hz=50):
        x, y, z = self.table_to_robot(vx, vy, vz)

        ee_frame = 'mallet_link'

        rot_z = self.compute_target_orientation(x, y, z, 'world', ee_frame)
        rot = np.array([
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])
        target_orientation = rot @ rot_z
        q_target = quaternion_from_matrix(target_orientation.tolist())

        target_stamped = PoseStamped()
        target_stamped.header.frame_id = 'panda_link0'
        target_stamped.pose.position.x = x
        target_stamped.pose.position.y = y
        target_stamped.pose.position.z = z
        target_stamped.pose.orientation.x = q_target[0]
        target_stamped.pose.orientation.y = q_target[1]
        target_stamped.pose.orientation.z = q_target[2]
        target_stamped.pose.orientation.w = q_target[3]

        # Attempt to read current EE pose via TF
        start_pos = (x, y, z)
        start_q = q_target
        tf = None
        try:
            tf = TransformListener()
            tf.waitForTransform(target_stamped.header.frame_id, ee_frame, rospy.Time(), rospy.Duration(1.0))
            trans, rotf = tf.lookupTransform(target_stamped.header.frame_id, ee_frame, rospy.Time(0))
            start_pos = (trans[0], trans[1], trans[2])
            start_q = (rotf[0], rotf[1], rotf[2], rotf[3])
        except Exception:
            # if TF fails, start from the target (controller will get a single jump)
            rospy.logwarn("TF lookup failed for ee frame; publishing direct target (no interpolation).")

        # compute motion duration from linear/angular speeds
        linear_distance = _vec_dist(start_pos, (x, y, z))
        dot = abs(start_q[0]*q_target[0] + start_q[1]*q_target[1] + start_q[2]*q_target[2] + start_q[3]*q_target[3])
        dot = min(1.0, max(-1.0, dot))
        angular_distance = 2.0 * math.acos(dot)  # angle between quaternions

        t_linear = linear_distance / linear_speed if linear_speed and linear_distance > 1e-6 else 0.0
        t_angular = angular_distance / angular_speed if angular_speed and angular_distance > 1e-6 else 0.0
        total_time = max(t_linear, t_angular, 0.0)

        # if both speeds are zero or TF failed and we chose fallback, publish once and exit
        if total_time == 0.0:
            target_stamped.header.stamp = rospy.Time.now()
            self.pose_pub.publish(target_stamped)
            if wait_robot:
                rospy.sleep(0.05)
            return True

        steps = max(1, int(round(total_time * rate_hz)))
        rate = rospy.Rate(rate_hz)
        for i in range(1, steps + 1):
            alpha = float(i) / float(steps)
            px = start_pos[0] + alpha * (x - start_pos[0])
            py = start_pos[1] + alpha * (y - start_pos[1])
            pz = start_pos[2] + alpha * (z - start_pos[2])
            q_interp = quaternion_slerp(start_q, q_target, alpha)

            target_stamped.header.stamp = rospy.Time.now()
            target_stamped.pose.position.x = px
            target_stamped.pose.position.y = py
            target_stamped.pose.position.z = pz
            target_stamped.pose.orientation.x = q_interp[0]
            target_stamped.pose.orientation.y = q_interp[1]
            target_stamped.pose.orientation.z = q_interp[2]
            target_stamped.pose.orientation.w = q_interp[3]

            self.pose_pub.publish(target_stamped)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break

        # ensure final pose sent a few times
        for _ in range(3):
            target_stamped.header.stamp = rospy.Time.now()
            self.pose_pub.publish(target_stamped)
            rospy.sleep(0.01)

        if wait_robot:
            rospy.sleep(0.05)
        return True

    def move_to_point(self, vx, vy, vz=0.1, wait_robot=False):
        print(f"    vx: {vx}, vy:{vy}, vz:{vz} ANGOLO")
        x, y, z = self.table_to_robot(vx, vy, vz)
        #print(f"    vx: {x}, vy:{y}, vz:{z} WORLD")

        """rot = [
            [-0.5, 0, 0.866, 0],
            [0, 1, 0, 0],
            [-0.866, 0, -0.5, 0],
            [0, 0, 0, 1]
        ]"""

        rot_z = self.compute_target_orientation(x, y, z, 'world', 'mallet_link')

        rot = np.array([
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        target_orientation = rot @ rot_z

        target_stamped = PoseStamped()
        target_stamped.header.frame_id = ''
        target_stamped.header.stamp = rospy.Time.now()

        target_stamped.pose.position.x = x
        target_stamped.pose.position.y = y
        target_stamped.pose.position.z = z

        # Convert rotation matrix to quaternion
        q = quaternion_from_matrix(target_orientation.tolist())
        target_stamped.pose.orientation.x = q[0]
        target_stamped.pose.orientation.y = q[1]
        target_stamped.pose.orientation.z = q[2]
        target_stamped.pose.orientation.w = q[3]

        print(f"target: {target_stamped.pose}")

        rate = rospy.Rate(50)
        while True:
            target_stamped.header.stamp = rospy.Time.now()
            # 3. Publish the Pose to the controller
            # This is the single, instantaneous command sent to the real-time controller.
            self.pose_pub.publish(target_stamped)
            print("Pose published")
            rate.sleep()
        
        # If wait_robot is True, you may want to add a small sleep to let the controller act
        #if wait_robot:
        #    rospy.sleep(0.01) 
        
        return True # Always returns success as publishing is instantaneous
    
    def move_to_point_prof(self, vx, vy, vz=0.1, wait_robot=False):

        return

class TargetVisualizer:
    def __init__(self, frame_id="world"):
        # Latched publisher so the sphere persists in RViz without requiring continuous republishing
        self.pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1, latch=True)
        self.frame_id = frame_id

    def publish_sphere(self, vx, vy, vz, diameter=0.06, rgba=(1.0, 0.0, 0.0, 0.9), ns="target", mid=0, frame_id='world'):

        x, y, z = PandaArm.table_to_robot(vx, vy, vz)

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
