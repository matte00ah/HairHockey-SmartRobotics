#!/usr/bin/env python3
import cv2
import numpy as np
import os
import math
import yaml
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(script_dir, "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

camera_topic = config["camera_topic"]
l_white = config["lower_white"]
u_white = config["upper_white"]

TABLE_WIDTH_CM = config["table_width_m"] * 100
TABLE_HEIGHT_CM = config["table_height_m"] * 100

LOWER_RED1 = np.array(config["lower_red1"], dtype=np.uint8)
UPPER_RED1 = np.array(config["upper_red1"], dtype=np.uint8)
LOWER_RED2 = np.array(config["lower_red2"], dtype=np.uint8)
UPPER_RED2 = np.array(config["upper_red2"], dtype=np.uint8)

# Funzione per trovare intersezione di due linee (Ax+By=C forma)
def line_intersection(l1, l2):
    # Appiattisci eventuali array annidati come [[x1, y1, x2, y2]]
    l1 = np.array(l1).flatten()
    l2 = np.array(l2).flatten()

    print(f"Calcolo intersezione tra linee: {l1} e {l2}")
    
    x1, y1, x2, y2 = map(float, l1)
    x3, y3, x4, y4 = map(float, l2)
    
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = A1 * x1 + B1 * y1
    
    A2 = y4 - y3
    B2 = x3 - x4
    C2 = A2 * x3 + B2 * y3
    
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-6:
        return None
    
    x = (B2 * C1 - B1 * C2) / det
    y = (A1 * C2 - A2 * C1) / det
    return int(x), int(y)

import cv2
import numpy as np

def apply_white_red_mask(frame):
    """
    Applica una maschera che isola aree bianche e rosse nell'immagine.
    Mostra e salva la maschera risultante.
    """

    # --- Maschera bianca ---
    """lower_white = np.array([150, 150, 150])
    upper_white = np.array([255, 255, 255])
    mask_white = cv2.inRange(frame, lower_white, upper_white)"""

    # --- Converti l'immagine in HSV ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Maschera bianca in HSV ---
    mask_white = cv2.inRange(hsv, l_white, u_white)

    mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # --- Combina bianco + rosso ---
    combined_mask = cv2.bitwise_or(mask_white, mask_red)

    # --- Applica la maschera all'immagine originale ---
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)

    # --- Mostra e salva ---
    cv2.imshow("Maschera bianca + rossa", combined_mask)
    cv2.imwrite("mask_white_red.png", combined_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


def filter_lines_by_red_hsv(lines, frame_bgr, red_ratio_threshold=0.3):
    """
    Elimina le linee che passano per aree rosse nel frame.
    
    lines: lista di linee (output di cv2.HoughLinesP)
    frame_bgr: immagine originale in formato BGR
    red_ratio_threshold: percentuale massima di pixel rossi ammessa
                         (0.3 = 30% di pixel rossi lungo la linea)
    """
    # Conversione in HSV per rilevare meglio il rosso
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    print("\n=== HSV medi per ogni linea trovata ===")
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        
        # Maschera per la linea
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        # Calcolo media HSV usando solo i pixel della linea
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]  # (H, S, V)
        H, S, V = mean_hsv
        #print(f"Linea {i+1}: H={H:.2f}, S={S:.2f}, V={V:.2f}")

    # Due range per il rosso in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Maschera per la linea
        mask = np.zeros(red_mask.shape, dtype=np.uint8)
        cv2.line(mask, (x1, y1), (x2, y2), 255, 3)
        
        # Percentuale di pixel rossi lungo la linea
        total_pixels = np.count_nonzero(mask)
        red_pixels = np.count_nonzero(cv2.bitwise_and(red_mask, mask))
        if total_pixels == 0:
            continue
        
        red_ratio = red_pixels / total_pixels

        # Se meno del 30% della linea è rossa → la teniamo
        if red_ratio < red_ratio_threshold:
            filtered.append(line)

    print(f"Linee totali: {len(lines)} | Linee mantenute: {len(filtered)}")
    return filtered

def process_frame(msg):
    print(f"Frame acquisito: {msg.shape}")
    frame = msg

    white_area = apply_white_red_mask(frame)

    gray = cv2.cvtColor(white_area, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Rilevamento bordi
    edges = cv2.Canny(blur, 50, 150)

    # Rilevamento linee con Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=40)
    
    debug_img = frame.copy()
    color = (0, 255, 0)  # verde
    thickness = 2

    for line_data in lines:
        x1, y1, x2, y2 = line_data[0]
        cv2.line(debug_img, (x1, y1), (x2, y2), color, thickness)
    cv2.imshow("Linee subito dopo Hough", debug_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if lines is None:
        print("⚠️ Nessuna linea trovata")
        return None, None, None
    print(f"Linee totali trovate: {len(lines)}")
    # Filtra linee che passano per aree rosse
    lines = filter_lines_by_red_hsv(lines, frame)
    print(f"Linee dopo filtro rosso: {len(lines)}")

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

    # Crea una copia dell'immagine per non sovrascrivere l'originale
    debug_img = frame.copy()

    # Colore e spessore delle linee
    color = (0, 255, 0)  # verde
    thickness = 2

    # Disegna tutte le linee trovate
    for line_data in best_lines:
        x1, y1, x2, y2 = line_data[0]
        cv2.line(debug_img, (x1, y1), (x2, y2), color, thickness)

            # Mostra il risultato
    cv2.imshow("Linee prima del filtro", debug_img)

    # Attendi un tasto per chiudere la finestra (0 = infinito)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


   # --- Selezione delle linee principali (4 lati tavolo) ---
    vertical_lines = []
    horizontal_lines = []

    for line_data in best_lines:
        line, ang_coef, q, length = line_data
        x1, y1, x2, y2 = line
        print(f"angolo=", ang_coef)
        if ang_coef > 0.8:
            vertical_lines.append(line_data)
        elif ang_coef < 0.1:
            horizontal_lines.append(line_data)

    print(f"Linee verticali trovate: {len(vertical_lines)}")
    print(f"Linee orizzontali trovate: {len(horizontal_lines)}")

    if len(vertical_lines) >= 2:
        vertical_lines.sort(key=lambda l: (l[0][0] + l[0][2]) / 2)
        left_line = vertical_lines[0]
        right_line = vertical_lines[-1]
    else:
        print("⚠️ Non abbastanza linee verticali trovate")
        return None, None, None

    if len(horizontal_lines) >= 2:
        horizontal_lines.sort(key=lambda l: (l[0][1] + l[0][3]) / 2)
        top_line = horizontal_lines[0]
        bottom_line = horizontal_lines[-1]
    else:
        print("⚠️ Non abbastanza linee orizzontali trovate")
        return None, None, None

    final_lines = [left_line, right_line, top_line, bottom_line]

    print("\n=== Linee principali selezionate ===")
    print(f"Sinistra: {left_line[0]}")
    print(f"Destra:   {right_line[0]}")
    print(f"Alto:     {top_line[0]}")
    print(f"Basso:    {bottom_line[0]}")

    # Visualizza le 4 linee principali
    debug_img = frame.copy()
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255)]
    for (line_data, color) in zip(final_lines, colors):
        x1, y1, x2, y2 = line_data[0]
        cv2.line(debug_img, (x1, y1), (x2, y2), color, 3)
    cv2.imshow("Linee principali (4 lati tavolo)", debug_img)
    cv2.imwrite("linee_principali.png", debug_img)

    cv2.waitKey(0)

    # calcola intersezioni
    intersections = []
    lines = final_lines
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

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title("Corner ordinati + Centro tavolo")
    plt.axis("off")
    plt.savefig("corner_e_centro.png")
    plt.show()

    return ordered_corners


if __name__ == "__main__":
    pass
