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
    arm.set_max_velocity_scaling_factor(1.0)      # Default: 0.1 → 10% della velocità massima
    arm.set_max_acceleration_scaling_factor(1.0)  # Default: 0.1 → 10% dell’accelerazione massima


    target_pose = Pose()
    #target_pose.header.frame_id = arm.get_planning_frame()
    target_pose.position.x = y - 0.425 #metà del tavolo
    target_pose.position.y = x - 0.97 #metà del tavolo
    target_pose.position.z = 1.0
    target_pose.orientation.w = 1.0  # orientazione neutra

    arm.set_start_state_to_current_state()
    arm.set_pose_target(target_pose)
    plan = arm.go(wait=True)

    arm.stop()
    arm.clear_pose_targets()


if __name__ == "__main__":
    # Esempio: chiama la funzione con valori di test
    move_to_point(0.2, 0.5, 1.0)  #coordinate robot ("0 1.2 0.7")
