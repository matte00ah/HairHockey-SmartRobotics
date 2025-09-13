#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
from montecarlo_filter import MontecarloFilter
from origin_detector import process_frame

script_dir = os.path.dirname(os.path.realpath(__file__))  # cartella dello script
config_path = os.path.join(script_dir, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

l_red1 = config["lower_red1"]
u_red1 = config["upper_red1"]
l_red2 = config["lower_red2"]
u_red2 = config["upper_red2"]
Y_MAX = config["table_height_m"]
camera_topic = config["camera_topic"]



def pixel_to_world(pt, corner1, corner2):
    # Corner in pixel
    p0 = np.array(corner1, dtype=np.float32)  # es. angolo in alto a sinistra
    p1 = np.array(corner2, dtype=np.float32)  # es. angolo in basso a sinistra

    # Assi del tavolo
    y_axis = p1 - p0
    y_len = np.linalg.norm(y_axis)
    y_axis /= y_len  # normalizza

    # Asse x = perpendicolare
    x_axis = np.array([y_axis[1], -y_axis[0]])

    # Trasforma punto pixel -> coord tavolo
    pt_vec = np.array(pt, dtype=np.float32) - p0
    wx = np.dot(pt_vec, x_axis) / np.linalg.norm(x_axis) / y_len * Y_MAX  # scala rispetto a Y_MAX
    wy = np.dot(pt_vec, y_axis) / y_len * Y_MAX

    return wx, wy, p0, x_axis, y_axis, y_len

def draw_axes(frame, p0, x_axis, y_axis, y_len, scale=0.5):
    """Disegna assi locali del tavolo su immagine"""
    p0 = tuple(p0.astype(int))

    # Asse X in rosso
    pX = (p0[0] + int(x_axis[0]*y_len*scale),
          p0[1] + int(x_axis[1]*y_len*scale))
    cv2.arrowedLine(frame, p0, pX, (0,0,255), 2, tipLength=0.2)

    # Asse Y in verde
    pY = (p0[0] + int(y_axis[0]*y_len*scale),
          p0[1] + int(y_axis[1]*y_len*scale))
    cv2.arrowedLine(frame, p0, pY, (0,255,0), 2, tipLength=0.2)

    # Origine in blu
    cv2.circle(frame, p0, 6, (255,0,0), -1)

    return frame

class DiskTracker:
    def __init__(self):
        # Nodo ROS
        rospy.init_node("disk_tracker")

        msg = rospy.wait_for_message(camera_topic, Image)
        corners = process_frame(msg)
        self.corner1 = corners[0]
        self.corner2 = corners[1]

        # CvBridge per convertire i messaggi ROS in immagini OpenCV
        self.bridge = CvBridge()

        # Sottoscrizione al topic della camera
        self.image_topic = rospy.get_param("~image_topic", "/image_raw")
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Parametri colore rosso in HSV
        self.lower_red1 = np.array(l_red1)
        self.upper_red1 = np.array(u_red1)
        self.lower_red2 = np.array(l_red2)
        self.upper_red2 = np.array(u_red2)

        # Kernel per pulizia mask
        self.kernel = np.ones((5,5), np.uint8)
        
        self.montecarlo = MontecarloFilter()

        rospy.loginfo("Red Disk Tracker avviato su topic: %s", self.image_topic)
        rospy.spin()


    def image_callback(self, msg):
        # Converti ROS Image in OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Converti in HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crea mask per il rosso
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Pulizia della mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Trova contorni
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Prendi il contorno più grande (disco)
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])                

                # Stampa posizione
                rospy.loginfo("Posizione disco (pixel): x=%d y=%d", cx, cy)
                # Disegna centro e contorno
                cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
                cv2.drawContours(frame, [c], -1, (0,255,0), 2)

                wx, wy, p0, x_axis, y_axis, y_len = pixel_to_world((cx, cy), self.corner1, self.corner2)
                rospy.loginfo("Disco in tavolo: X=%.2f m, Y=%.2f m", wx, wy)

                # Disegno assi sulla frame
                frame = draw_axes(frame, p0, x_axis, y_axis, y_len)

                self.montecarlo.run(wx, wy)

        # Mostra immagine e mask
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        tracker = DiskTracker()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
