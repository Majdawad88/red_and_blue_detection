#git clone https://github.com/Majdawad88/red_and_blue_detection.git

import cv2
import numpy as np
import time
import sys
from picamera2 import Picamera2

# --- Configuration ---
# Define the lower boundary for the first range of Red in HSV (H: 0-10)
lower_red1 = np.array([0, 120, 70])
# Define the upper boundary for the first range of Red in HSV
upper_red1 = np.array([10, 255, 255])
# Define the lower boundary for the second range of Red (wraps around the spectrum H: 170-180)
lower_red2 = np.array([170, 120, 70])
# Define the upper boundary for the second range of Red in HSV
upper_red2 = np.array([180, 255, 255])
# Define the lower boundary for Blue in HSV (Hue roughly 100-140)
lower_blue = np.array([100, 150, 0])
# Define the upper boundary for Blue in HSV
upper_blue = np.array([140, 255, 255])

# Set the dimensions for a single side (half) of the combined output window
HALF_W, HALF_H = 560, 420
# Calculate the total window width (double the half width) and total height
DISPLAY_W, DISPLAY_H = HALF_W * 2, HALF_H

# --- State ---
# Initialize the starting detection mode to "RED"
mode = "RED"
# Global flag to track if the "QUIT" button has been clicked via mouse
quit_clicked = False
# Global flag to track if the "MODE" toggle button has been clicked via mouse
mode_clicked = False

# Function to handle mouse click events on the CV2 window
def mouse_callback(event, x, y, flags, param):
    # Access global variables to update state based on user interaction
    global quit_clicked, mode_clicked
    # Check if the left mouse button was pressed down
    if event == cv2.EVENT_LBUTTONDOWN:
        # If click coordinates are within the "QUIT" button box (10-110px wide, 10-50px high)
        if 10 <= x <= 110 and 10 <= y <= 50:
            quit_clicked = True # Set the flag to exit the program
        # If click coordinates are within the "MODE" button box (130-260px wide, 10-50px high)
        if 130 <= x <= 260 and 10 <= y <= 50:
            mode_clicked = True # Set the flag to switch color modes

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
        # Capture the raw image data from the camera sensor
        frame_rgb = picam2.capture_array()
        # Convert the raw RGB frame to BGR format for OpenCV compatibility
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Flip the image vertically to correct the camera orientation
        frame_bgr = cv2.flip(frame_bgr, 0)

        # Apply a Gaussian blur to smooth the image and reduce high-frequency noise
        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        # Convert the blurred BGR image to the HSV color space for easier color filtering
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1. Choose Mask
        if mode == "RED":
            # Create a mask for the lower red hue range
            m1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # Create a mask for the upper red hue range
            m2 = cv2.inRange(hsv, lower_red2, upper_red2)
            # Combine both red masks into a single binary image
            mask = cv2.bitwise_or(m1, m2)
            # Set the UI text color theme to Red (BGR: 0, 0, 255)
            color_theme = (0, 0, 255)
        else:
            # Create a mask for the blue hue range
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            # Set the UI text color theme to Blue (BGR: 255, 0, 0)
            color_theme = (255, 0, 0)

        # --- NEW: Contour Detection & Rectangle ---
        # Detect the boundaries of the white shapes in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Identify the single largest detected shape based on its area
            largest_contour = max(contours, key=cv2.contourArea)

            # Process the shape only if it is large enough to be a valid target
            if cv2.contourArea(largest_contour) > 500:
                # Get the coordinates and size for a bounding box around the shape
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Draw a green rectangle on the live frame around the object
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Display the name of the current target mode above the box
                cv2.putText(frame_bgr, f"Target {mode}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Draw UI
        # Draw the red background box for the QUIT button
        cv2.rectangle(frame_bgr, (10, 10), (110, 50), (0, 0, 200), -1)
        # Overlay the "QUIT" text on the button
        cv2.putText(frame_bgr, "QUIT", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Draw the green background box for the MODE button
        cv2.rectangle(frame_bgr, (130, 10), (260, 50), (0, 200, 0), -1)
        # Overlay the "MODE" text on the button
        cv2.putText(frame_bgr, "MODE", (155, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Display the current active color mode status on the screen
        cv2.putText(frame_bgr, f"CURRENT: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color_theme, 2)

        # 3. Handle Logic for Switching
        if mode_clicked:
            # Toggle the mode variable between "RED" and "BLUE"
            mode = "BLUE" if mode == "RED" else "RED"
            # Reset the click flag to wait for the next interaction
            mode_clicked = False

        # 4. Prepare Display
        # Resize the live tracking frame for the side-by-side view
        left_img = cv2.resize(frame_bgr, (HALF_W, HALF_H))
        # Convert the 1-channel mask to 3-channels and resize it for display
        right_img = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (HALF_W, HALF_H))
        # Join the live feed and the mask horizontally into one window
        combined = np.hstack((left_img, right_img))

        # Show the final combined output in the created window
        cv2.imshow(win_name, combined)

        # Exit the loop if 'q' is pressed or the QUIT button is clicked
        if (cv2.waitKey(1) & 0xFF == ord('q')) or quit_clicked:
            break

finally:
    # Safely turn off the camera hardware
    picam2.stop()
    # Close all OpenCV display windows to clean up system resources
    cv2.destroyAllWindows()
