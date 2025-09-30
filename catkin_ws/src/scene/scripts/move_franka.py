#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose


def move_to_point(x, y, z):
    """
    Muove il braccio Franka verso la posizione (x, y, z).
    """
    moveit_commander.roscpp_initialize(sys.argv)
    #rospy.init_node('panda_move', anonymous=True)

    arm = moveit_commander.MoveGroupCommander("arm_group")

    target_pose = Pose()
    target_pose.position.x = x
    target_pose.position.y = y
    target_pose.position.z = z
    target_pose.orientation.w = 1.0  # orientazione neutra

    arm.set_start_state_to_current_state()
    arm.set_pose_target(target_pose)
    plan = arm.go(wait=True)

    arm.stop()
    arm.clear_pose_targets()


if __name__ == "__main__":
    # Esempio: chiama la funzione con valori di test
    move_to_point(0.3, 0.6, 0.2)
