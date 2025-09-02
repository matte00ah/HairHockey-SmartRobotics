#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
from table_width_detector import apply_white_mask

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

camera_topic = config["camera_topic"]
canny_min = config["canny_threshold_min"]
canny_max = config["canny_threshold_max"]
hough_threshold = config["hough_threshold"]
hough_min_length = config["hough_min_length"]
hough_max_gap = config["hough_max_gap"]

# Funzione per trovare intersezione di due linee (Ax+By=C forma)
def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1*x1 + B1*y1
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2*x3 + B2*y3
    
    det = A1*B2 - A2*B1
    if det == 0:
        return None
    x = (B2*C1 - B1*C2) / det
    y = (A1*C2 - A2*C1) / det
    return int(x), int(y)

def process_frame(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    white_area = apply_white_mask(frame)

    gray = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Rilevamento bordi
    edges = cv2.Canny(blur, 50, 150)

    # Rilevamento linee con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=hough_min_length, maxLineGap=hough_max_gap)

    # Disegna linee trovate
    line_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calcola tutte le intersezioni
    intersections = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i][0], lines[j][0])
            if pt is not None:
                x, y = pt
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    intersections.append(pt)

    # Convex hull per trovare i 4 punti estremi
    pts = np.array(intersections)
    hull = cv2.convexHull(pts)
    if len(hull) > 4:
        # prendo i 4 punti estremi del bounding box
        x, y, w, h = cv2.boundingRect(hull)
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    else:
        corners = hull.reshape(-1, 2).tolist()

    # Disegno corner
    for (x, y) in corners:
        cv2.circle(line_img, (x, y), 8, (0, 0, 255), -1)

    print("Corner trovati:")
    for i, c in enumerate(corners):
        print(f"Corner {i+1}: {c}")

    origin_world = min(corners, key=lambda p: p[0] + p[1])
    print(f"Corner selezionato come nostro WORLD: {origin_world} ")
    cv2.circle(line_img, (origin_world[0], origin_world[1]), 10, (255, 0, 0), -1)

    cv2.imshow("Corner del tavolo", line_img)
    cv2.waitKey(0)  # aspetta pressione di un tasto
    cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("table_width_once", anonymous=True)

    # Prende UN SOLO frame dal topic
    msg = rospy.wait_for_message(camera_topic, Image)
    process_frame(msg)