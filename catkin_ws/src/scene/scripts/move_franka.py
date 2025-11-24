#!/usr/bin/env python3
import sys
import rospy
import moveit_commander
import time
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import Marker
import argparse
from moveit_msgs.msg import OrientationConstraint, Constraints
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler, quaternion_from_matrix

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
        default=1.7,
        help="Target X coordinate in table frame (float).",
    )
    parser.add_argument(
        "-y",
        "--pos_y",
        type=float,
        default=0.425,
        help="Target Y coordinate in table frame (float).",
    )
    parser.add_argument(
        "-z",
        "--pos_z",
        type=float,
        default=0.0,
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
        # Inizializza ROS e MoveIt UNA VOLTA
        moveit_commander.roscpp_initialize(sys.argv)
        self.arm = moveit_commander.MoveGroupCommander("arm_group")
        self.arm.set_max_velocity_scaling_factor(1.0)
        self.arm.set_max_acceleration_scaling_factor(1.0)
        self.arm.set_pose_reference_frame('world')
        self.frame_id = frame_id
        print(f"Robot reference frame: {self.arm.get_planning_frame()}")

    def robot_alignment(self):
        """
        Impone che l'asse x dell'end-effector sia allineato con l'asse -y del frame world.
        Permette rotazione libera attorno a x, vincola y e z.
        """
        rot = [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        q = quaternion_from_matrix(rot)

        oc = OrientationConstraint()
        oc.header.frame_id = self.arm.get_planning_frame()
        oc.link_name = self.arm.get_end_effector_link()
        oc.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        # Vincoli stretti su tutti gli assi: orientazione fissa
        oc.absolute_x_axis_tolerance = 0.05
        oc.absolute_y_axis_tolerance = 0.05
        oc.absolute_z_axis_tolerance = 0.05
        oc.weight = 1.0
        constraints = Constraints()
        constraints.orientation_constraints = [oc]
        self.arm.set_path_constraints(constraints)

    def clear_constraints(self):
        self.arm.clear_path_constraints()

    @staticmethod
    def table_to_world_transform(x, y, z):
        """Applies the same transformation used for motion to map table coords to robot coords."""
        rx = y - 0.425
        ry = x - 0.97
        rz = z + 0.82 + 0.15 #z + <altezza_tavolo> + <distanza punta stick e base paddle>
        return rx, ry, rz
    
    def move_to_point(self, x, y, z=1.0):
        #target_pose = PoseStamped()
        
        rot = [
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        q = quaternion_from_matrix(rot)

        target_pose = Pose()

        # Trasformazione da tavolo a robot
        target_pose.position.x = x
        target_pose.position.y = y
        target_pose.position.z = z

        # Quaternion valido per paddle verticale
        target_pose.orientation.x = q[0]
        target_pose.orientation.y = q[1]
        target_pose.orientation.z = q[2]
        target_pose.orientation.w = q[3]

        # target_pose.header.frame_id = self.frame_id
        # target_pose.pose = pose

        self.arm.set_start_state_to_current_state()
        #print(time.perf_counter())
        self.arm.set_pose_target(target_pose)
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        #print(time.perf_counter())
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

    vx, vy, vz = robot.table_to_world_transform(args.pos_x, args.pos_y, args.pos_z)

    print(f"X:{vx} - Y:{vy} - Z:{vz}")

    visualizer.publish_sphere(vx, vy, vz, diameter=args.sphere_diameter)


    # Imposta il vincolo di orientazione: x_ee allineato con -y_world
    robot.robot_alignment()
    success = robot.move_to_point(vx, vy, vz)
    robot.clear_constraints()

    moveit_commander.roscpp_shutdown()
