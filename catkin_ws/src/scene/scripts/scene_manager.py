#!/usr/bin/env python3

import rospy
import rospkg
import os
from moveit_commander import roscpp_initialize, PlanningSceneInterface
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelStates

class SceneManager:
    def __init__(self):
        roscpp_initialize([])
        rospy.init_node("scene_manager")

        self.rospack = rospkg.RosPack()
        self.scene_pkg_path = self.rospack.get_path("scene")

        rospy.loginfo("Waiting for /get_planning_scene service...")
        rospy.wait_for_service('/get_planning_scene')
        rospy.sleep(1)

        self.scene = PlanningSceneInterface(synchronous=True)
        rospy.sleep(2)  # aspetta che MoveIt sia pronto

        # aggiungi oggetti statici
        self.add_static_objects()

        # sottoscrivi il puck dinamico
        rospy.Subscriber("/gazebo/model_states", ModelStates, self.update_puck)
        
        rospy.loginfo("SceneManager pronto")
        rospy.spin()

    def create_pose(self, x, y, z, qx=0, qy=0, qz=0, qw=1, frame="world"):
        ps = PoseStamped()
        ps.header.frame_id = frame
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def add_static_objects(self):
        table_path = os.path.join(self.scene_pkg_path, "models/plane/meshes/airhockey-plane.stl")
        borders_path = os.path.join(self.scene_pkg_path, "models/table_borders/meshes/airhockey-borders.stl")
        puck_path = os.path.join(self.scene_pkg_path, "models/puck/meshes/airhockey-puck.stl")


        # Tavolo
        table_pose = self.create_pose(0.0, 0.0, 0.0)
        self.scene.add_mesh("table", table_pose, table_path)
        
        borders_pose = self.create_pose(0.0, 0.0, 0.0)
        self.scene.add_mesh("table_borders", borders_pose, borders_path)
        
        puck_pose = self.create_pose(0.0, 0.3, 0.9)
        self.scene.add_mesh( "puck", puck_pose, puck_path, size=(0.001, 0.001, 0.001))

        rospy.loginfo("Oggetti statici aggiunti alla planning scene")

    def update_puck(self, msg):
        try:
            idx = msg.name.index("puck")
        except ValueError:
            return

        puck_pose_gazebo = msg.pose[idx]

        puck_pose = PoseStamped()
        puck_pose.header.frame_id = "world"
        puck_pose.pose = puck_pose_gazebo

        # rimuove e riaggiunge la mesh con scala corretta
        self.scene.remove_world_object("puck")
        puck_path = os.path.join(self.scene_pkg_path, "models/puck/meshes/airhockey-puck.stl")
        self.scene.add_mesh("puck", puck_pose, puck_path, size=(0.001, 0.001, 0.001))


if __name__ == "__main__":
    SceneManager()
