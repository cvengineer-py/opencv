# Exno.3 – Implementation of Various Segmentation Algorithms
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Load Image ---
img = cv2.imread('data/peacock.jpeg')
if img is None:
    print("Image not found!")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 1. Simple Thresholding ---
_, thresh_simple = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# --- 2. Adaptive Thresholding ---
thresh_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# --- 3. Otsu’s Thresholding ---
_, thresh_otsu = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)



# --- Display Results ---
titles = [
    'Original', 'Simple Threshold', 'Adaptive Threshold',
    'Otsu Threshold'
]
images = [
    img_rgb, thresh_simple, thresh_adapt,
    thresh_otsu
]

plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray' if i in [1, 2, 3] else None)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
