#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
from montecarlo_filter import MontecarloFilter
from origin_detector_new import process_frame

script_dir = os.path.dirname(os.path.realpath(__file__))  
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

# HSV bounds (uint8, più veloce)
LOWER_RED1 = np.array(config["lower_red1"], dtype=np.uint8)
UPPER_RED1 = np.array(config["upper_red1"], dtype=np.uint8)
LOWER_RED2 = np.array(config["lower_red2"], dtype=np.uint8)
UPPER_RED2 = np.array(config["upper_red2"], dtype=np.uint8)

Y_MAX = config["table_height_m"]
X_MAX = config["table_width_m"]
camera_topic = config["camera_topic"]

def compute_homography(ordered_corners):
    """Precalcola matrice omografia una sola volta."""
    real_corners = np.array([
        [0, 0],
        [X_MAX, 0],
        [X_MAX, Y_MAX],
        [0, Y_MAX]
    ], dtype=np.float32)
    pixel_corners = np.array(ordered_corners, dtype=np.float32)
    H, _ = cv2.findHomography(pixel_corners, real_corners)
    return H

def pixel_to_world(pt, H):
    """Converte pixel in coordinate reali usando omografia precalcolata."""
    px = np.array([[pt]], dtype=np.float32)  # (1,1,2)
    world_pt = cv2.perspectiveTransform(px, H)
    return world_pt[0][0]

class DiskTracker:
    def __init__(self):
        rospy.init_node("disk_tracker")

        # Calcolo corner e omografia solo una volta
        msg = rospy.wait_for_message(camera_topic, Image)
        self.corners = process_frame(msg)
        self.H = compute_homography(self.corners)

        self.bridge = CvBridge()
        self.image_topic = rospy.get_param("~image_topic", "/image_raw")
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

        self.kernel = np.ones((5, 5), np.uint8)
        self.montecarlo = MontecarloFilter()

        rospy.loginfo("Red Disk Tracker avviato su topic: %s", self.image_topic)
        rospy.spin()

    def image_callback(self, msg):
        # Conversione immagine
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # HSV + mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1) | cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)

        # Pulizia
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Usa connected components (più veloce di findContours per centri)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

        if num_labels > 1:
            # Prendi il blob più grande (escludendo background 0)
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cx, cy = centroids[largest_idx]

            # Coordinate mondo
            wx, wy = pixel_to_world((cx, cy), self.H)
            
            # Debug limitato (ogni 10 frame)
            rospy.loginfo("Disco: pixel=(%.0f,%.0f), world=(%.2f, %.2f m)", cx, cy, wx, wy)

            self.montecarlo.run(wx, wy)

        # Se vuoi debug con GUI (molto costoso, lascia commentato)
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(1)

if __name__ == "__main__":
    try:
        tracker = DiskTracker()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()

