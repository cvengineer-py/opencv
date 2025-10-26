import cv2
import numpy as np

# --- Load image ---
image_path = r'data\road.jpeg'
image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not found or path is incorrect")

# --- Grayscale and blur ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --- Edge detection ---
edges = cv2.Canny(blurred, 50, 150)

# --- Region of interest mask ---
height, width = edges.shape
mask = np.zeros_like(edges)
roi = np.array([[
    (0, height),
    (width*0.1, height*0.5),
    (width*0.9, height*0.5),
    (width, height)
]], dtype=np.int32)
cv2.fillPoly(mask, roi, 255)
masked_edges = cv2.bitwise_and(edges, mask)

# --- Hough Line Transform ---
lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)

line_image = np.copy(image)
left_lines, right_lines = [], []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            if slope < -0.2:
                left_lines.append(line)
            elif slope > 0.2:
                right_lines.append(line)

# --- Draw lane lines ---
for line in left_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
for line in right_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# --- Overlay on original image ---
result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# --- Display ---
cv2.imshow('Original', image)
cv2.imshow('Road Margins', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Save result ---
cv2.imwrite('road_with_margins.jpg', result)
