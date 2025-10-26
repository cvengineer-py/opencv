# Exno.2 – Implementation of Histogram
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread(r'data/peacock.jpeg')
if img is None:
    print("Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Grayscale Histogram
hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.plot(hist_gray, color='k')
plt.show()

# 2. Color Histogram (B, G, R)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.xlim([0, 256])
plt.show()

# 3. Histogram Equalization (Gray Image)
eq_gray = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([eq_gray], [0], None, [256], [0, 256])

cv2.imshow('Original Grayscale', gray)
cv2.imshow('Equalized Grayscale', eq_gray)

plt.figure()
plt.title('Equalized Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.plot(hist_eq, color='k')
plt.xlim([0, 256])
plt.show()

# 4. 2D Histogram (Hue vs Saturation)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist_2d = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])

plt.figure()
plt.title('2D Hue-Saturation Histogram')
plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.imshow(hist_2d, interpolation='nearest')
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
