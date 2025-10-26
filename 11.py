import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- Paths ----
left_image_path = r"data\face.jpeg"
right_image_path = r"data\car.jpeg"

# ---- Load images in grayscale ----
left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

# Ensure both images are loaded
if left_img is None or right_img is None:
    raise FileNotFoundError("Check image paths.")

# Resize right image if needed
if left_img.shape != right_img.shape:
    right_img = cv2.resize(right_img, (left_img.shape[1], left_img.shape[0]))

# ---- StereoSGBM parameters ----
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16,
    blockSize=5
)

# ---- Compute disparity map ----
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Normalize for visualization
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ---- Display results ----
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("Left Image")
plt.imshow(left_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Right Image")
plt.imshow(right_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Disparity Map")
plt.imshow(disp_norm, cmap='plasma')
plt.axis('off')

plt.tight_layout()
plt.show()
