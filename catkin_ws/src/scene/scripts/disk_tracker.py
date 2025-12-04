#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os
import glob
from queue import Queue, Empty
import threading
import time
import matplotlib.pyplot as plt
from montecarlo_filter import MontecarloFilter
from origin_detector import process_frame
from move_franka import PandaArm
import subprocess
import glob
import re

script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

LOWER_RED1 = np.array(config["lower_red1"], dtype=np.uint8)
UPPER_RED1 = np.array(config["upper_red1"], dtype=np.uint8)
LOWER_RED2 = np.array(config["lower_red2"], dtype=np.uint8)
UPPER_RED2 = np.array(config["upper_red2"], dtype=np.uint8)


DISK_LOWER_RED1_LEFT = np.array(config["disk_lower_red1_left"], dtype=np.uint8)
DISK_UPPER_RED1_LEFT = np.array(config["disk_upper_red1_left"], dtype=np.uint8)
DISK_LOWER_RED2_LEFT = np.array(config["disk_lower_red2_left"], dtype=np.uint8)
DISK_UPPER_RED2_LEFT = np.array(config["disk_upper_red2_left"], dtype=np.uint8)
DISK_LOWER_RED2 = np.array(config["disk_lower_red2"], dtype=np.uint8)
DISK_UPPER_RED2 = np.array(config["disk_upper_red2"], dtype=np.uint8)

Y_MAX = config["table_height_m"]
X_MAX = config["table_width_m"]

#CAMERA_TOPIC = config["camera_topic"]
FIND_CORNERS = config["find_corners"]
GAME_POSE = config["game_pose"]

CORNER_1 = config["corner_1"]
CORNER_2 = config["corner_2"]
CORNER_3 = config["corner_3"]
CORNER_4 = config["corner_4"]

TRAINING = False
TRAINING_ROUNDS = 1800

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

def pixel_to_meter_fast(pt, H):
    x, y = pt
    den = H[2,0]*x + H[2,1]*y + H[2,2]
    wx = (H[0,0]*x + H[0,1]*y + H[0,2]) / den
    wy = (H[1,0]*x + H[1,1]*y + H[1,2]) / den
    return wx, wy

def barrel_dist_correction(src):
    width  = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4,1),np.float64)

    k1 = -1.0e-6; # negative to remove barrel distortion
    k2 = 0.0
    p1 = 0
    p2 = 0

    distCoeff[0,0] = k1
    distCoeff[1,0] = k2
    distCoeff[2,0] = p1
    distCoeff[3,0] = p2

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 2.1        # define focal length x
    cam[1,1] = 2.1        # define focal length y

    # here the undistortion will be computed
    dst = cv2.undistort(src,cam,distCoeff)
    return dst

def get_video_devices():
    """Ritorna la lista dei device video es: ['/dev/video0', '/dev/video1']"""
    return sorted(glob.glob("/dev/video*"))

def open_first_free_camera():
    devices = get_video_devices()
    print("Trovate camere:", devices)
    """if len(devices) > 1:
        dev = devices[1]"""
    for dev in devices:
        print(f"\n[INFO] Controllo {dev}")

        print(f"[INFO] Provo ad aprire {dev}…")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if cap.isOpened():
            print(f"[SUCCESS] Camera aperta: {dev}")
            return cap

        print(f"[ERROR] Impossibile aprire {dev}, passo al successivo.")
        cap.release()

    print("[ERROR] Nessuna camera disponibile.")
    return None


def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_img = param
        pixel = hsv_img[y, x]
        print(f"POS ({x},{y}) → H:{pixel[0]} S:{pixel[1]} V:{pixel[2]}")

class DiskTracker:
    def __init__(self):
        print("Init")
        #cam = open_first_free_camera()
        #print(f"cam: {cam}")
        self.cap = open_first_free_camera()
        try:
            if not self.cap.isOpened():
                return
            
            for i in range(10):
                _, _ = self.cap.read()

            if FIND_CORNERS:
                # Extract game table corners
                ret, frame = self.cap.read()
                if not ret:
                    return
                
                #frame = barrel_dist_correction(frame)
                # --- Dividi immagine stereo ZED 2i in sinistra e destra ---
                h, w, _ = frame.shape
                left_img = frame[:, :w//2]
                frame = left_img
                frame = barrel_dist_correction(frame)

                print(f"shape: {frame.shape}")
                
                cv2.imshow("Frame per angoli",frame)
                if (cv2.waitKey(0) & 0xFF) == ord('q'):  # TODO: mettere waitKey(1) per avere video
                    pass

                self.corners = process_frame(frame)

            else:
                self.corners = [CORNER_1, CORNER_2, CORNER_3, CORNER_4]
            
            self.kernel = np.ones((3, 3), np.uint8)

            robot = PandaArm()

            print(f"Move to Game pose... {GAME_POSE[0]}")
            robot.move_to_point(vx=GAME_POSE[0], vy=GAME_POSE[1], vz=GAME_POSE[2], wait_robot=True)

            self.H = compute_homography(self.corners)

            self.montecarlo = MontecarloFilter(robot=robot)

            # Coda con un solo slot per frame più recente
            self.frame_queue = Queue(maxsize=1)

            self.counter = 0

            self.processing_loop()
            self.cap.release()
        except KeyboardInterrupt:
            self.cap.release()
        finally:
            self.cap_release()
            cv2.destroyAllWindows()
            
        # Thread di elaborazione
        """self.processing_thread = threading.Thread(target=self.processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.camera_frame()"""

    """def camera_callback(self, msg):
        #print("Chiamata camera_callback", time.perf_counter())
        if not self.frame_queue.empty():
            _ = self.frame_queue.get_nowait()  # rimuove frame vecchio
        self.frame_queue.put_nowait(msg)"""

    """def camera_frame(self):
        print("Lettura frame attivato")
        while self.cap.isOpened():
            try:
                #frame = self.frame_queue.get(timeout=0.1)
                ret, frame = self.cap.read()

                if not self.frame_queue.empty():
                    _ = self.frame_queue.get_nowait()  # rimuove frame vecchio
                self.frame_queue.put_nowait(frame)
            except:
                print("Ex")
                continue  # nessun frame disponibile"""

    def processing_loop(self):
        """Thread che elabora continuamente il frame più recente""" 
        #while self.cap.isOpened():
        while True:
            try:
                for _ in range(10):
                    self.cap.grab()
                
                ret, frame = self.cap.retrieve()
                """if not self.cap.grab():  # prova a scartare frame successivi
                    break
                #frame = self.frame_queue.get(timeout=0.1)
                ret, frame = self.cap.read()
                if not self.cap.grab():  # prova a scartare frame successivi
                    break"""

                #frame = barrel_dist_correction(frame)

                """cv2.imshow("Tracking dischi", frame)
                if cv2.waitKey(1) == ord('q'):
                    break"""
            except:
                print("Ex")
                continue  # nessun frame disponibile
            
            h, w, _ = frame.shape
            left_img = frame[:, :w//2]
            frame = left_img
            frame = barrel_dist_correction(frame)

            # HSV + mask
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv_h, hsv_w, _ = frame.shape

            """cv2.imshow("Image", frame)
            cv2.setMouseCallback("Image", show_hsv, hsv)
            if cv2.waitKey(0) == ord('q'):
                break"""
            
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            # Maschera rossa con apertura e chiusura
            #mask[:, :hsv_w//2] = cv2.inRange(hsv[:, :hsv_w//2], DISK_LOWER_RED1_LEFT, DISK_UPPER_RED1_LEFT)
            mask1_left = cv2.inRange(hsv[:, :hsv_w//2], DISK_LOWER_RED1_LEFT, DISK_UPPER_RED1_LEFT)
            mask2_left = cv2.inRange(hsv[:, :hsv_w//2], DISK_LOWER_RED2_LEFT, DISK_UPPER_RED2_LEFT)
            mask[:, :hsv_w//2] = cv2.bitwise_or(mask1_left, mask2_left)

            mask[:, hsv_w//2:] = cv2.inRange(hsv[:, hsv_w//2:], DISK_LOWER_RED2, DISK_UPPER_RED2)

            #mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
            #mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
            #mask = cv2.bitwise_and(mask)

            # Morfologia
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

            """cv2.imshow("Tracking dischi", frame)
            if cv2.waitKey(0) == ord('q'):
                break
            cv2.imshow("Tracking dischi", mask)
            if cv2.waitKey(0) == ord('q'):
                break"""

            # Connected components per centri
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            #print(f"Num etichette: {num_labels}")
            cerchi = []
            area_nuova = []
            if num_labels > 1:
                for i in range(1, num_labels):
                    cx, cy = centroids[i]
                    area = stats[i, cv2.CC_STAT_AREA]
                    
                    # Filtra piccoli disturbi (opzionale)
                    if area < 25:
                        #print("area minima")
                        continue
                    else:
                        cerchi.append([cx,cy])
                        area_nuova.append(stats[i, cv2.CC_STAT_AREA])

                #print(f"Len: {len(area_nuova)}")
                if len(area_nuova) > 0:
                    #min_idx = 1 + np.argmin(area_nuova[1:])  # indice del cerchio più piccolo
                    min_idx = np.argmin(area_nuova)  # indice del cerchio più piccolo
                    cx, cy = cerchi[min_idx]
                    #print(f"Disco trovato: pixel=({cx:.0f},{cy:.0f}))")
                    cv2.circle(frame, (int(cx), int(cy)), 10, (0, 255, 0), 2)   # contorno verde
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)   # punto rosso al centro
                    """cv2.imshow("Tracking dischi", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break"""
                    print(f"\n\nDISCO: measurement in pixel {cx, cy}")
                    wx, wy = pixel_to_meter_fast((cx, cy), self.H)
                    if TRAINING: 
                        self.montecarlo.training(wx, wy)
                        self.counter += 1
                    else: 
                        self.montecarlo.run(wx, wy)   
            else:
                if not TRAINING:
                    # disco non trovato → None
                    self.montecarlo.run(None, None)
                    continue
            
            if self.counter >= TRAINING_ROUNDS and TRAINING:
                # Per salvare il file di allenamento
                self.montecarlo.save_filter_state()
                break

            #time.sleep(3)  # piccola pausa per non saturare la CPU

if __name__ == "__main__":
    try:
        tracker = DiskTracker()
    except:
        pass
    cv2.destroyAllWindows()