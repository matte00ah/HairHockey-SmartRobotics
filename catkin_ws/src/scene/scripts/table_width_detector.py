#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import math
import json
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

script_dir = os.path.dirname(os.path.realpath(__file__))  # cartella dello script
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

l_white = config["lower_white"]
u_white = config["upper_white"]
table_height_m = config["table_height_m"]
table_width_m = config["table_width_m"]
camera_topic = config["camera_topic"]
canny_min = config["canny_threshold_min"]
canny_max = config["canny_threshold_max"]
hough_threshold = config["hough_threshold"]
hough_min_length = config["hough_min_length"]
hough_max_gap = config["hough_max_gap"]

# Funzione per calcolare i coefficienti della retta ax + by + c = 0
def line_coeff(x1, y1, x2, y2):
    a = y2 - y1
    b = x1 - x2
    c = x2*y1 - x1*y2
    return a, b, c

# Controlla se due rette sono parallele
def are_parallel(a1, b1, a2, b2, tol=1e-2):
    return abs(a1*b2 - a2*b1) < tol

# Distanza perpendicolare tra due rette parallele
def parallel_distance(a, b, c1, c2):
    return abs(c2 - c1) / math.sqrt(a**2 + b**2)

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

    # Applica la maschera
    white_area = apply_white_mask(frame)

    # Converti in grayscale e trova i bordi
    gray = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_min, canny_max)

    # Trova i contorni
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=hough_min_length, maxLineGap=hough_max_gap)
    
    if lines is None:
        rospy.loginfo("Nessuna linea rilevata")
        return

    # Calcola coefficienti delle linee e salva con coordinate
    lines_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a, b, c = line_coeff(x1, y1, x2, y2)
        lines_data.append({'coords': (x1, y1, x2, y2), 'a': a, 'b': b, 'c': c})
        print('coords: ' + str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2) +
      '  a:' + str(a) + ' b:' + str(b) + ' c:' + str(c))
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Trova la coppia di linee parallele più distanti
    max_dist = 0
    best_pair = None
    for i in range(len(lines_data)):
        for j in range(i+1, len(lines_data)):
            l1 = lines_data[i]
            l2 = lines_data[j]
            if are_parallel(l1['a'], l1['b'], l2['a'], l2['b'], tol=100):
                dist = parallel_distance(l1['a'], l1['b'], l1['c'], l2['c'])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (l1['coords'], l2['coords'])

    if best_pair is not None:
        rospy.loginfo(f"Larghezza visibile tavolo in pixel: {int(max_dist)} px")

        pixel_size = table_height_m / int(max_dist)  # metri per pixel
        rospy.loginfo(f"Dimensione di un pixel: {pixel_size*1000:.2f} mm")

        # Disegna le due linee selezionate in rosso
        for coords in best_pair:
            x1, y1, x2, y2 = coords
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mostra risultati (opzionale)
    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("table_width_once", anonymous=True)

    # Prende UN SOLO frame dal topic
    msg = rospy.wait_for_message(camera_topic, Image)
    process_frame(msg)