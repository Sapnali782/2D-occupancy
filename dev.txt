import cv2
import numpy as np
import matplotlib.pyplot as plt

# Constants for grid and image processing
GRID_RESOLUTION = 0.1  
# meters per grid cell
GRID_SIZE_X = 100      
# grid cells in X direction (10 meters)
GRID_SIZE_Y = 100      
# grid cells in Y direction (10 meters)

# Initialize occupancy grid
occupancy_grid = np.zeros((GRID_SIZE_Y, GRID_SIZE_X), dtype=np.uint8)

# Function to process a frame and update the occupancy grid
def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold image to create binary map (occupied vs free)
    _, binary_map = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Resize binary map to match grid size
    resized_map = cv2.resize(binary_map, (GRID_SIZE_X, GRID_SIZE_Y))
    
    # Update occupancy grid based on resized binary map
    for y in range(GRID_SIZE_Y):
        for x in range(GRID_SIZE_X):
            if resized_map[y, x] > 0:
                occupancy_grid[y, x] = 1  # occupied
            else:
                occupancy_grid[y, x] = 0  # free

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or replace with your camera ID

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a single frame
ret, frame = cap.read()
if not ret:
    print("Error: Failed to capture image.")
else:
    # Process the captured frame
    process_frame(frame)
    
    # Display the frame
    cv2.imshow('Camera Feed', frame)
    cv2.waitKey(1)  # Display the image for a short time

    # Display the occupancy grid
    plt.figure(figsize=(8, 8))
    plt.imshow(occupancy_grid, cmap='gray', origin='lower')
    plt.colorbar(label='Occupancy')
    plt.title('Occupancy Grid Map')
    plt.xlabel('Grid cells (X direction)')
    plt.ylabel('Grid cells (Y direction)')
    plt.grid(True)
    plt.show()

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
