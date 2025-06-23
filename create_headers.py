import cv2
import numpy as np
import os

# Ensure the directory exists
os.makedirs('Hand Tracking Project/Header', exist_ok=True)

# Define colors and filenames
headers = [
    ((0, 0, 255), 'red.png'),    # Red (BGR)
    ((255, 0, 0), 'blue.png'),   # Blue (BGR)
    ((0, 255, 0), 'green.png'),  # Green (BGR)
    ((0, 0, 0), 'black.png'),    # Black (BGR)
]

for color, filename in headers:
    img = np.zeros((125, 200, 3), np.uint8)
    img[:] = color
    cv2.imwrite(f'Hand Tracking Project/Header/{filename}', img)

print('Header images created!') 