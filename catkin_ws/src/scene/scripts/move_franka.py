#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import time
import os
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker
import argparse
from tf.transformations import quaternion_from_matrix
import yaml

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

X = config["table_width_m"] / 2
Y = config["table_height_m"] / 2
Z = config["z"]

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
        default=0,
        help="Target X coordinate in table frame (float).",
    )
    parser.add_argument(
        "-y",
        "--pos_y",
        type=float,
        default=0.6,
        help="Target Y coordinate in table frame (float).",
    )
    parser.add_argument(
        "-z",
        "--pos_z",
        type=float,
        default=1,
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
        self.arm = moveit_commander.MoveGroupCommander("arm_group")
        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)
        self.arm.set_pose_reference_frame('world')
        self.frame_id = frame_id

        self.arm.set_goal_position_tolerance(0.01)  # default 0.001
        self.arm.set_goal_orientation_tolerance(0.01)
        self.arm.set_goal_joint_tolerance(0.01)

        print(f"Robot reference frame: {self.arm.get_planning_frame()}")

    @staticmethod
    def table_to_world_transform(x, y, z):
        """Applies the same transformation used for motion to map table coords to robot coords."""
        rx = Y - y
        ry = X - x
        rz = z + Z # z + <altezza_tavolo>
        return rx, ry, rz
    
    def move_to_point(self, vx, vy, vz=0.1, wait_robot=False):
        print(f"vx: {vx}, vy:{vy}, vz:{vz} ANGOLO")
        x, y, z = self.table_to_world_transform(vx, vy, vz)
        print(f"vx: {x}, vy:{y}, vz:{z} WORLD")

        rot = [
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
            [0, 0, 1, 0],
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
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        #print(time.perf_counter())
        print("Posizione raggiunta!!!")
        return success
    
class TargetVisualizer:
    def __init__(self, frame_id="world"):
        # Latched publisher so the sphere persists in RViz without requiring continuous republishing
        self.pub = rospy.Publisher("/visualization_marker", Marker, queue_size=1, latch=True)
        self.frame_id = frame_id

    def publish_sphere(self, x, y, z, diameter=0.06, rgba=(1.0, 0.0, 0.0, 0.9), ns="target", mid=0, frame_id='world'):
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
    vx, vy, vz = 1.57, 0.425, 0  # rispetto all'angolo
    pvx, pvy, pvz = robot.table_to_world_transform(vx, vy, vz)
    visualizer.publish_sphere(pvx, pvy, pvz, diameter=args.sphere_diameter)

    # Imposta il vincolo di orientazione: x_ee allineato con -y_world
    success = robot.move_to_point(vx, vy, vz)

    moveit_commander.roscpp_shutdown()
