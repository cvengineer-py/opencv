import cv2
import numpy as np

# Load image in grayscale
img = cv2.imread('data\star.jpeg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Cannot load image 'star.jpg'")

# Edge detection
edges = cv2.Canny(img, 50, 150)

# Corner detection
corners_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
corners = cv2.goodFeaturesToTrack(img, 200, 0.01, 10)
if corners is not None:
    for x, y in np.intp(corners).reshape(-1, 2):
        cv2.circle(corners_img, (x, y), 3, (0, 255, 0), -1)

# Line detection
lines_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0 + 1000*(-b)), int(y0 + 1000*(a))
        x2, y2 = int(x0 - 1000*(-b)), int(y0 - 1000*(a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Display results
cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Corners", corners_img)
cv2.imshow("Lines", lines_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
