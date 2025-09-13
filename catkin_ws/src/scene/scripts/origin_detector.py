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
    # Converti in HSV per filtrare il bianco
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pix = hsv.reshape(-1, 3)

    pixels_tuples = [tuple(p) for p in pix]

    from collections import Counter
    counts = Counter(pixels_tuples)
    most_common_pixel, freq = counts.most_common(1)[0]

    print(f"Pixel HSV più frequente sul tavolo: {most_common_pixel} (presente {freq} volte)")

    # Range del bianco (modifica se necessario)
    lower_white = np.array(l_white)
    upper_white = np.array(u_white)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    # num_white_pixels = cv2.countNonZero(mask)
    # print(f"Numero di pixel bianchi nella maschera: {num_white_pixels}")

    # Applica la maschera
    white_area = cv2.bitwise_and(frame, frame, mask=mask)

    return white_area

def process_frame(msg):
    bridge = CvBridge()
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    print(frame.shape)
    
    #MODIFICA PER PRENDERE FOTO INVECE CHE FRAME ROS
    #frame = cv2.imread("c:\\Users\\Matteo Bulgarelli\\Downloads\\1.jpg")
    #frame = cv2.resize(frame, (640, 480))

    white_area = apply_white_mask(frame)

    gray = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Rilevamento bordi
    edges = cv2.Canny(blur, 50, 150)
    print(edges)

    # Rilevamento linee con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=30) #max=10

    best_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1-x2)>10:  # non è verticale
            m = (y2 - y1) / (x2 - x1)
            angle = math.atan(m)  # angolo in radianti tra -pi/2 e pi/2
            ang_coef = abs(angle) / (math.pi / 2)
            q = y1 - m * x1
        else:
            # retta perfettamente verticale
            ang_coef = 1
            q = x1  # per rette verticali, uso l'intercetta x
        
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # controllo se linea è migliore di altra in best_lines
        if len(best_lines) > 0:
            cnt = 0
            for b_line in best_lines:
                if abs(b_line[1] - ang_coef) > 0.1 or abs(b_line[2] - q) > 10:  # coefficiente angolare simile e intercetta simile
                    cnt+=1
                    continue
                else:    
                    # rette sono molto simili, quindi si prende quella con lunghezza maggiore
                    if b_line[3] > length:
                        break
                    else:
                        b_line = [line[0], ang_coef, q, length]
                        break
            if cnt == len(best_lines):
                best_lines.append([line[0], ang_coef, q, length])

        else:
            best_lines.append([line[0], ang_coef, q, length])

    #DEBUG
    print("numero linee trovate : ", len(best_lines))
    
    # trova retta verticale e quelle orizzontali
    best_v = -1
    v_line = None
    for i in range(len(best_lines)):
        if best_lines[i][1] > best_v:
            best_v = best_lines[i][1]
            v_line = i
    
    v_line = best_lines.pop(v_line)
    h_lines = best_lines

    # calcola punti di intersezione per trovare corner esatti
    corners = []
    if v_line is not None and h_lines is not None:
        for line in h_lines:
            pt = line_intersection(v_line[0], line[0])
            if pt is not None:
                corners.append(pt)

    lines = h_lines
    lines.append(v_line)

    # Disegna linee trovate
    line_img = frame.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calcola tutte le intersezioni
    intersections = []
    if lines is not None and len(lines) >= 2:
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

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    cv2.imshow("Edges", edges)
    plt.title("Corner del tavolo (da intersezioni linee)")
    plt.axis("off")
    plt.show()
    
    return corners


if __name__ == "__main__":
    rospy.init_node("origin", anonymous=True)

    # Prende UN SOLO frame dal topic
    msg = rospy.wait_for_message(camera_topic, Image)
    process_frame(msg)
    #MODIFICA per funzionare con foto
    #process_frame(None)