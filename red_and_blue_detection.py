# Clone the red and blue detection repository from GitHub
# !git clone https://github.com/Majdawad88/red_and_blue_detection.git

import cv2 # Import OpenCV library for image processing
import numpy as np # Import NumPy for numerical and array operations
import time # Import time for handling delays and sleep
import sys # Import sys for system-specific parameters and functions
from picamera2 import Picamera2 # Import the Picamera2 library to control the Raspberry Pi camera

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
win_name = "Color Tracker" # Define the name of the display window
cv2.namedWindow(win_name) # Create the named window for output
cv2.setMouseCallback(win_name, mouse_callback) # Link the mouse callback function to this window

# --- Camera Setup ---
picam2 = Picamera2() # Initialize the Picamera2 instance
picam2.configure("preview") # Set the camera configuration for a live preview mode
picam2.start() # Start the camera hardware stream

try:
    while True: # Begin the main processing loop
        # Capture a live frame from the camera as an RGB array
        frame_rgb = picam2.capture_array()
        # Convert frame from RGB to BGR for OpenCV compatibility
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # Flip the frame vertically to fix camera orientation
        frame_bgr = cv2.flip(frame_bgr, 0)

        # Pre-process for better detection
        # Apply Gaussian blur to smooth the image and reduce high-frequency noise
        blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)
        # Convert the blurred BGR image to HSV color space for easier filtering
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 1. Choose Mask
        if mode == "RED": # Logic for Red detection
            # Filter HSV for the first red range
            m1 = cv2.inRange(hsv, lower_red1, upper_red1)
            # Filter HSV for the second red range
            m2 = cv2.inRange(hsv, lower_red2, upper_red2)
            # Combine both red masks into one binary image
            mask = cv2.bitwise_or(m1, m2)
            # Set UI color theme to Red (BGR format)
            color_theme = (0, 0, 255)
        else: # Logic for Blue detection
            # Filter HSV for the blue range
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            # Set UI color theme to Blue (BGR format)
            color_theme = (255, 0, 0)

        # --- NEW: Contour Detection & Rectangle ---
        # Find all white blobs (foreground) in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours: # Check if any blobs were found
            # Identify the single largest detected shape based on its area
            largest_contour = max(contours, key=cv2.contourArea)

            # Process the shape only if it is large enough to be a valid target (ignores noise)
            if cv2.contourArea(largest_contour) > 500:
                # Get the coordinates (x,y) and size (w,h) for a bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                # Draw a green rectangle on the live BGR frame around the object
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
                # Display the name of the current target mode above the box
                cv2.putText(frame_bgr, f"Target {mode}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 2. Draw UI
        # Draw the red background rectangle for the QUIT button
        cv2.rectangle(frame_bgr, (10, 10), (110, 50), (0, 0, 200), -1)
        # Overlay the "QUIT" text onto the red button
        cv2.putText(frame_bgr, "QUIT", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # Draw the green background rectangle for the MODE button
        cv2.rectangle(frame_bgr, (130, 10), (260, 50), (0, 200, 0), -1)
        # Overlay the "MODE" text onto the green button
        cv2.putText(frame_bgr, "MODE", (155, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Display the current active color mode status below the buttons
        cv2.putText(frame_bgr, f"CURRENT: {mode}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color_theme, 2)

        # 3. Handle Logic for Switching
        if mode_clicked: # If the MODE button flag was set by mouse click
            # Toggle the mode variable between "RED" and "BLUE"
            mode = "BLUE" if mode == "RED" else "RED"
            # Reset the click flag to wait for the next interaction
            mode_clicked = False

        # 4. Prepare Display
        # Resize the live tracking frame for the half-screen view
        left_img = cv2.resize(frame_bgr, (HALF_W, HALF_H))
        # Convert the 1-channel mask to 3-channels so it can be combined with BGR images
        right_img = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (HALF_W, HALF_H))
        # Join the live feed and the mask horizontally into one large frame
        combined = np.hstack((left_img, right_img))

        # Show the final combined image in the tracking window
        cv2.imshow(win_name, combined)

        # Check if the 'q' key is pressed or the QUIT flag is True to exit
        if (cv2.waitKey(1) & 0xFF == ord('q')) or quit_clicked:
            break

finally:
    # Safely power down the camera hardware
    picam2.stop()
    # Close all OpenCV GUI windows to free system resources
    cv2.destroyAllWindows()
