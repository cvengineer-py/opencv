import cv2
import numpy as np
from matplotlib import pyplot as plt

test_image = cv2.imread(r'data\test\face.jpeg')
known_images = [cv2.imread(r'data\known\face.jpeg') for _ in range(1)]


if test_image is None or any(img is None for img in known_images):
    raise FileNotFoundError("Check that all images exist in the folder.")

# Calculate color histogram for an image
def calc_hist(img):
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], 
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

# Recognize face by comparing histograms
test_hist = calc_hist(test_image)
distances = [cv2.compareHist(test_hist, calc_hist(kimg), cv2.HISTCMP_BHATTACHARYYA)
             for kimg in known_images]

recognized_index = np.argmin(distances)
if distances[recognized_index] <= 0.5:
    print(f"Face recognized as person {recognized_index + 1}")
else:
    print("Face not recognized")

# Display images
plt.figure(figsize=(10, 5))
plt.subplot(1, len(known_images) + 1, 1)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title('Test Image')

for i, kimg in enumerate(known_images):
    plt.subplot(1, len(known_images) + 1, i + 2)
    plt.imshow(cv2.cvtColor(kimg, cv2.COLOR_BGR2RGB))
    plt.title(f'Known Image {i + 1}')

plt.tight_layout()
plt.show()
