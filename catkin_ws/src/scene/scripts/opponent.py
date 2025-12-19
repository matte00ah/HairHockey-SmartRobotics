#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Vector3

rospy.init_node("push_disk")

def main():
    apply_wrench = rospy.ServiceProxy("/gazebo/apply_body_wrench", ApplyBodyWrench)
    wrench = Wrench(force=Vector3(-0.2, -0.55, 0), torque=Vector3(0, 0, 0))

    apply_wrench(
        body_name="puck::puck_link",
        reference_frame="world",
        wrench=wrench,
        start_time=rospy.Time(0.1),
        duration=rospy.Duration(0.1)
    )

if __name__ == "__main__":
    main()
