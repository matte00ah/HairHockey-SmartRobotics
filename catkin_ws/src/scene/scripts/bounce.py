#!/usr/bin/env python3

import rospy
import time
import numpy as np
from stl import mesh
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Twist
from scipy.spatial import ConvexHull
from shapely.ops import nearest_points
# pip install numpy-stl shapely

STL_PATH = "/home/user/Documents/GitHub/HairHockey/catkin_ws/src/scene/models/table_borders/meshes/airhockey-borders.stl"
MODEL_NAME = "puck"

current_x = current_y = current_z = 0.0
current_vx = current_vy = 0.0


def model_state_callback(msg):
    global current_x, current_y, current_z
    global current_vx, current_vy
    if MODEL_NAME in msg.name:
        idx = msg.name.index(MODEL_NAME)
        pose = msg.pose[idx]
        twist = msg.twist[idx]

        current_x = pose.position.x
        current_y = pose.position.y
        current_z = pose.position.z

        current_vx = twist.linear.x
        current_vy = twist.linear.y
        
def load_table_outline(stl_path):
    """Estrae il contorno XY della mesh STL"""
    m = mesh.Mesh.from_file(stl_path)
    points_2d = np.column_stack((m.x.flatten(), m.y.flatten()))
    hull = ConvexHull(points_2d)
    polygon = Polygon(points_2d[hull.vertices])
    return polygon

def move_disk():
    has_collided = False

    table_outline = load_table_outline(STL_PATH)
    rospy.loginfo("Caricato contorno del tavolo STL.")

    rospy.init_node("bouncing_disk_curved", anonymous=True)
    rospy.Subscriber("/gazebo/model_states", ModelStates, model_state_callback)
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
    
    rate = rospy.Rate(60)
    rospy.loginfo("Aspetto i dati di /gazebo/model_states...")
    rospy.sleep(0.5)  # un attimo per ricevere i primi dati

    while not rospy.is_shutdown():
        x, y, z = current_x, current_y, current_z
        vx, vy = current_vx, current_vy

        # dt = 1 / 800
        # aggiorna posizione  TODO prendere le posizioni predette da montyecarlo
        # x += vx / dt
        # y += vy / dt

        disk_point = Point(x, y, z)

        if has_collided and table_outline.buffer(-0.16).contains(disk_point):
            has_collided = False

        # se esce dal contorno, rimbalza
        if not has_collided and not table_outline.buffer(-0.15).contains(disk_point):  # buffer negativo = margine interno
            print("MURO!!")
            
            inner_outline = table_outline.buffer(-0.1)
            nearest_point = nearest_points(disk_point, inner_outline.exterior)[1]
            print(f"nearest: {nearest_point.x}, {nearest_point.y}")
            
            print(f"current vel: {vx} {vy}")
            print(f"current pos: {x} {y}")

            normal_vec = np.array([x - nearest_point.x, y - nearest_point.y])
            if np.linalg.norm(normal_vec) != 0:
                normal_vec /= np.linalg.norm(normal_vec)
            
            vel = np.array([vx, vy])

            vel = vel - 2 * np.dot(vel, normal_vec) * normal_vec
            vx, vy = vel * 0.9  # perdita di energia 10%
            print(f"new vel: {vx} {vy}")

            dt = 1/60
            x += vx * dt
            y += vy * dt
            #x = x + 1 * (nearest_point.x - x)
            #y = y + 1 * (nearest_point.y - y)
            print(f"new pos: {x} {y}")

            has_collided = True

            # pubblica stato in gazebo
            state = ModelState()
            state.model_name = MODEL_NAME
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = z
            state.twist.linear.x = vx
            state.twist.linear.y = vy
            state.twist.linear.z = 0.0

            try:
                set_state(state)
            except rospy.ServiceException as e:
                rospy.logerr(e)        

        rate.sleep()

if __name__ == "__main__":
    move_disk()
