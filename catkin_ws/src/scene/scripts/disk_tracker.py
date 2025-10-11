#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
from queue import Queue, Empty
import threading
import time
import matplotlib.pyplot as plt
#from montecarlo_new import MontecarloFilter
from montecarlo_filter import MontecarloFilter
from origin_detector import process_frame
from move_franka import PandaArm

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

LOWER_RED1 = np.array(config["lower_red1"], dtype=np.uint8)
UPPER_RED1 = np.array(config["upper_red1"], dtype=np.uint8)
LOWER_RED2 = np.array(config["lower_red2"], dtype=np.uint8)
UPPER_RED2 = np.array(config["upper_red2"], dtype=np.uint8)

Y_MAX = config["table_height_m"]
X_MAX = config["table_width_m"]
camera_topic = config["camera_topic"]

def compute_homography(ordered_corners):
    real_corners = np.array([
        [0, 0],
        [X_MAX, 0],
        [X_MAX, Y_MAX],
        [0, Y_MAX]
    ], dtype=np.float32)
    pixel_corners = np.array(ordered_corners, dtype=np.float32)
    H, _ = cv2.findHomography(pixel_corners, real_corners)
    return H

def pixel_to_world_fast(pt, H):
    x, y = pt
    den = H[2,0]*x + H[2,1]*y + H[2,2]
    wx = (H[0,0]*x + H[0,1]*y + H[0,2]) / den
    wy = (H[1,0]*x + H[1,1]*y + H[1,2]) / den
    return wx, wy

class DiskTracker:
    def __init__(self):
        rospy.init_node("disk_tracker_threads")

        # Calcolo corner e omografia una sola volta
        msg = rospy.wait_for_message(camera_topic, Image)
        self.corners = process_frame(msg)

        #robot in posizione vicina al tavolo
        robot = PandaArm()
        robot.move_to_point(0.8 + 0.97, 0 + 0.425, 1.0)  #guarda move_franka.py

        self.H = compute_homography(self.corners)

        self.bridge = CvBridge()
        self.kernel = np.ones((3, 3), np.uint8)
        self.montecarlo = MontecarloFilter(robot=robot)

        # Coda con un solo slot per frame più recente
        self.frame_queue = Queue(maxsize=1)

        # Subscriber ROS veloce
        rospy.Subscriber(camera_topic, Image, self.camera_callback, queue_size=1)

        # Thread di elaborazione
        self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        rospy.loginfo("Disk Tracker avviato con thread e coda singolo slot")
        rospy.spin()

    def camera_callback(self, msg):
        #print("Chiamata camera_callback", time.perf_counter())
        if not self.frame_queue.empty():
            _ = self.frame_queue.get_nowait()  # rimuove frame vecchio
        self.frame_queue.put_nowait(msg)


    def processing_loop(self):
        """Thread che elabora continuamente il frame più recente"""
        while not rospy.is_shutdown():
            try:
                msg = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue  # nessun frame disponibile

            print("Chiamata processing_loop",time.perf_counter())
            
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            #print frame
            #plt.figure(figsize=(10, 6))
            #plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #plt.show()
            

            # HSV + mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1) | cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            # Connected components per centri
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            
            if num_labels > 1:
                largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                cx, cy = centroids[largest_idx]
                wx, wy = pixel_to_world_fast((cx, cy), self.H)
                rospy.loginfo("Disco: pixel=(%.0f,%.0f), world=(%.2f, %.2f m)", cx, cy, wx, wy)
                print("Chiamata montecarlo", time.perf_counter())
                self.montecarlo.run(wx, wy)
            else:
                # disco non trovato → None
                self.montecarlo.run(None, None)


if __name__ == "__main__":
    try:
        tracker = DiskTracker()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()