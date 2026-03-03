#git clone https://github.com/Majdawad88/red_and_blue_detection.git

import cv2
import numpy as np
import time
import sys
from picamera2 import Picamera2

# --- Configuration ---
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

HALF_W, HALF_H = 560, 420
DISPLAY_W, DISPLAY_H = HALF_W * 2, HALF_H

# --- State ---
mode = "RED"
quit_clicked = False
mode_clicked = False

def mouse_callback(event, x, y, flags, param):
    global quit_clicked, mode_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check buttons (coordinates relative to the resized window)
        if 10 <= x <= 110 and 10 <= y <= 50:
            quit_clicked = True
        if 130 <= x <= 260 and 10 <= y <= 50:
            mode_clicked = True

# --- Window Setup ---
win_name = "Color Tracker"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, mouse_callback)

# --- Camera Setup ---
picam2 = Picamera2()
picam2.configure("preview")
picam2.start()

try:
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.flip(frame_bgr, 0)

        # Pre-process for better detection
        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1. Choose Mask
        if mode == "RED":
            m1 = cv2.inRange(hsv, lower_red1, upper_red1)
            m2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(m1, m2)
            color_theme = (0, 0, 255) 
        else:
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            color_theme = (255, 0, 0)

        # --- NEW: Contour Detection & Rectangle ---
        # Find all white blobs in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest blob (by area)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Only draw if the blob is big enough (filter out noise)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Draw the rectangle around the detected object
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Add a label above the rectangle
                cv2.putText(frame_bgr, f"Target {mode}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Draw UI
        cv2.rectangle(frame_bgr, (10, 10), (110, 50), (0, 0, 200), -1)
        cv2.putText(frame_bgr, "QUIT", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        cv2.rectangle(frame_bgr, (130, 10), (260, 50), (0, 200, 0), -1)
        cv2.putText(frame_bgr, "MODE", (155, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame_bgr, f"CURRENT: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color_theme, 2)

        # 3. Handle Logic for Switching
        if mode_clicked:
            mode = "BLUE" if mode == "RED" else "RED"
            mode_clicked = False

        # 4. Prepare Display
        left_img = cv2.resize(frame_bgr, (HALF_W, HALF_H))
        right_img = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (HALF_W, HALF_H))
        combined = np.hstack((left_img, right_img))

        cv2.imshow(win_name, combined)

        if (cv2.waitKey(1) & 0xFF == ord('q')) or quit_clicked:
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
