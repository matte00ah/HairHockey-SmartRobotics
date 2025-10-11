#!/usr/bin/env python
import sys
import rospy
import moveit_commander
import time
from geometry_msgs.msg import Pose

class PandaArm:
    def __init__(self):
        # Inizializza ROS e MoveIt UNA VOLTA
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node('panda_move', anonymous=True)

        self.arm = moveit_commander.MoveGroupCommander("arm_group")
        self.arm.set_max_velocity_scaling_factor(1.0)
        self.arm.set_max_acceleration_scaling_factor(1.0)

    def move_to_point(self, x, y, z=1.0, wait_robot=False):
        self.arm.stop()
        self.arm.clear_pose_targets()
        target_pose = Pose()
        # Trasformazione da tavolo a robot(l'origine delle coordinate che escono da montecarlo è 
        # l'angolo del tavolo opposto al robot in alto, dobbiamo quindi riportarle al centro del 
        # tavolo per renderle compresibili dal robot. In più la x e la y sono invertite tra
        # tavolo(visto dalla telecamera) e robot)
        target_pose.position.x = y - 0.425
        target_pose.position.y = x - 0.97
        target_pose.position.z = z

        # Quaternion valido per paddle verticale
        target_pose.orientation.x = 0.0
        target_pose.orientation.y = 1.0
        target_pose.orientation.z = 0.0
        target_pose.orientation.w = 0.0

        self.arm.set_start_state_to_current_state()
        #print(time.perf_counter())
        self.arm.set_pose_target(target_pose)
        success = self.arm.go(wait=wait_robot)
        #self.arm.stop()
        #self.arm.clear_pose_targets()
        #print(time.perf_counter())
        return success


if __name__ == "__main__":
    # Esempio: chiama la funzione con valori di test
    robot = PandaArm()
    robot.move_to_point(0.8 + 0.97, 0 + 0.425, 1.0)  #coordinate robot ("0 1.2 0.7")
