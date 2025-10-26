# Exno.7 Medical Image Processing (No Functions)
import cv2
import matplotlib.pyplot as plt

# --- Load image ---
file_path = r'data\brain.jpeg'
img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Cannot load image from {file_path}")

# --- Apply Gaussian Blur ---
blurred = cv2.GaussianBlur(img, (5,5), 0)

# --- Apply Otsu's Thresholding ---
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# --- Find and draw contours ---
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contoured_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contoured_img, contours, -1, (0, 255, 0), 2)

# --- Display images ---
images = [img, blurred, thresh, contoured_img]
titles = ["Original Image", "Blurred Image", "Thresholded Image", "Contours"]

import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(1, len(images), i+1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()
