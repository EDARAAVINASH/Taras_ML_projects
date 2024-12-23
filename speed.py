import cv2
import numpy as np
from time import time, sleep

# Parameters
largura_min = 80  # Minimum width of the rectangle
altura_min = 80   # Minimum height of the rectangle
offset = 6        # Error margin in pixels
pos_linha_1 = 500  # Position of the first line
pos_linha_2 = 550  # Position of the second line
delay = 60        # FPS of the video
real_distance_m = 3  # Real-world distance between lines in meters

detec = []
carros = 0
timestamps = {}  # Store timestamps for vehicles
speeds = []  # Store speeds of detected vehicles

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    tempo = float(1 / delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the two lines for speed detection
    cv2.line(frame1, (25, pos_linha_1), (1200, pos_linha_1), (255, 127, 0), 3)
    cv2.line(frame1, (25, pos_linha_2), (1200, pos_linha_2), (255, 127, 0), 3)

    for (i, c) in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        for (cx, cy) in detec:
            # Detect crossing the first line
            if cy < (pos_linha_1 + offset) and cy > (pos_linha_1 - offset):
                timestamps[cx] = time()
                detec.remove((cx, cy))

            # Detect crossing the second line
            if cy < (pos_linha_2 + offset) and cy > (pos_linha_2 - offset):
                if cx in timestamps:
                    elapsed_time = time() - timestamps[cx]
                    speed = real_distance_m / elapsed_time * 3.6  # Convert m/s to km/h
                    speeds.append(speed)
                    carros += 1
                    cv2.line(frame1, (25, pos_linha_2), (1200, pos_linha_2), (0, 127, 255), 3)
                    detec.remove((cx, cy))
                    del timestamps[cx]
                    print(f"Car detected: {carros}, Speed: {speed:.2f} km/h")

    # Display vehicle count and average speed
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    cv2.putText(frame1, f"VEHICLE COUNT: {carros}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, f"AVG SPEED: {avg_speed:.2f} km/h", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
