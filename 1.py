# Exno.1 – Implementation of Various Filter Techniques
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read and prepare image
img_bgr = cv2.imread(r"data\peacock.jpeg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Kernel setup
kernel_size = 7
morph_kernel = np.ones((5, 5), np.uint8)

# Apply filters
avg_blur = cv2.blur(img_rgb, (kernel_size, kernel_size))
gaussian_blur = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), 0)
median_blur = cv2.medianBlur(img_rgb, kernel_size)
bilateral_filter = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)

sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1)
sobel_edges = np.sqrt(sobel_x**2 + sobel_y**2)


canny_edges = cv2.Canny(img_gray, 100, 200)
dilated_img = cv2.dilate(img_gray, morph_kernel, iterations=1)
eroded_img = cv2.erode(img_gray, morph_kernel, iterations=1)

# Display results
titles = [
    'Original', 'Averaging', 'Gaussian', 'Median',
    'Bilateral', 'Sobel', 'Canny', 'Dilation', 'Erosion'
]
images = [
    img_rgb, avg_blur, gaussian_blur, median_blur,
    bilateral_filter, sobel_edges, canny_edges, dilated_img, eroded_img
]

plt.figure(figsize=(15, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
