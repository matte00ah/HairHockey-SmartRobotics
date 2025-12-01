#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import os
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import json
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

camera_topic = config["camera_topic"]
l_white = config["lower_white"]
u_white = config["upper_white"]

# === Dimensioni reali tavolo (in cm, da modificare!) ===
TABLE_WIDTH_CM = 184  
TABLE_HEIGHT_CM = 75  

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

def apply_white_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range del bianco dal file di config
    lower_white = np.array(l_white)
    upper_white = np.array(u_white)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_area = cv2.bitwise_and(frame, frame, mask=mask)
    return white_area

def process_frame(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    print(f"Frame acquisito: {frame.shape}")

    white_area = apply_white_mask(frame)

    gray = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Rilevamento bordi
    edges = cv2.Canny(blur, 50, 150)

    # Rilevamento linee con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=30)

    best_lines = []
    if lines is None:
        print("⚠️ Nessuna linea trovata")
        return None, None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1-x2)>10:  # non è verticale
            m = (y2 - y1) / (x2 - x1)
            angle = math.atan(m)
            ang_coef = abs(angle) / (math.pi / 2)
            q = y1 - m * x1
        else:
            ang_coef = 1
            q = x1
        
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        if len(best_lines) > 0:
            cnt = 0
            for b_line in best_lines:
                if abs(b_line[1] - ang_coef) > 0.1 or abs(b_line[2] - q) > 10:
                    cnt+=1
                    continue
                else:    
                    if b_line[3] > length:
                        break
                    else:
                        b_line = [line[0], ang_coef, q, length]
                        break
            if cnt == len(best_lines):
                best_lines.append([line[0], ang_coef, q, length])
        else:
            best_lines.append([line[0], ang_coef, q, length])

    print("Numero linee trovate :", len(best_lines))
    
    if len(best_lines) < 2:
        print("⚠️ Non abbastanza linee trovate")
        return None, None, None

    # trova verticale e orizzontali
    best_v = -1
    v_line = None
    for i in range(len(best_lines)):
        if best_lines[i][1] > best_v:
            best_v = best_lines[i][1]
            v_line = i
    
    v_line = best_lines.pop(v_line)
    h_lines = best_lines

    # calcola intersezioni
    intersections = []
    lines = h_lines + [v_line]
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            pt = line_intersection(lines[i][0], lines[j][0])
            if pt is not None:
                x, y = pt
                if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                    intersections.append(pt)

    if len(intersections) < 4:
        print("⚠️ Non abbastanza intersezioni trovate")
        return None, None, None

    # Convex hull → 4 corner
    pts = np.array(intersections)
    hull = cv2.convexHull(pts)

    if len(hull) > 4:
        x, y, w, h = cv2.boundingRect(hull)
        corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    else:
        corners = hull.reshape(-1, 2).tolist()

    # Ordina i corner: [TL, TR, BR, BL]
    corners = sorted(corners, key=lambda p: (p[1], p[0]))
    top_points = sorted(corners[:2], key=lambda p: p[0])
    bottom_points = sorted(corners[2:], key=lambda p: p[0])
    ordered_corners = [top_points[0], top_points[1], bottom_points[1], bottom_points[0]]

    # Calcolo centro tavolo
    cx = int(sum([p[0] for p in ordered_corners]) / 4)
    cy = int(sum([p[1] for p in ordered_corners]) / 4)
    center_table = (cx, cy)

    print("Corner ordinati (TL, TR, BR, BL):")
    for i, c in enumerate(ordered_corners):
        print(f"Corner {i+1}: {c}")
    print(f"Centro tavolo (pixel): {center_table}")

    # Disegno per debug
    line_img = frame.copy()
    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
    for (corner, color) in zip(ordered_corners, colors):
        cv2.circle(line_img, tuple(corner), 8, color, -1)
    cv2.circle(line_img, center_table, 10, (0, 255, 255), -1)

    #plt.figure(figsize=(10, 6))
    #plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    #plt.title("Corner ordinati + Centro tavolo")
    #plt.axis("off")
    #plt.show()

    return ordered_corners


if __name__ == "__main__":
    rospy.init_node("origin", anonymous=True)

    # Prende UN SOLO frame dal topic
    msg = rospy.wait_for_message(camera_topic, Image)
    corners, center_px, center_cm = process_frame(msg)

    # Debug finale
    if corners is not None:
        print("==== RISULTATI ====")
        print(f"Corners: {corners}")
        print(f"Centro (pixel): {center_px}")
        print(f"Centro (cm): {center_cm}")
