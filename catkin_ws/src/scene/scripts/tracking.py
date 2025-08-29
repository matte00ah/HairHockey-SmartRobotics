#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class RedDiskTracker:
    def __init__(self):
        # Nodo ROS
        rospy.init_node("red_disk_tracker")

        # CvBridge per convertire i messaggi ROS in immagini OpenCV
        self.bridge = CvBridge()

        # Sottoscrizione al topic della camera
        self.image_topic = rospy.get_param("~image_topic", "/image_raw")
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

        # Parametri colore rosso in HSV
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([179, 255, 255])

        # Kernel per pulizia mask
        self.kernel = np.ones((5,5), np.uint8)

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

        # Mostra immagine e mask
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

if __name__ == "__main__":
    try:
        tracker = RedDiskTracker()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
