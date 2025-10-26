import cv2
from matplotlib import pyplot as plt

# ---- Paths ----
authorized_face_path = r'data\known\face.jpeg'  # Authorized face
test_image_path = r'data\test\face.jpeg'       # Test image

# ---- Load images ----
authorized_face_img = cv2.imread(authorized_face_path, cv2.IMREAD_GRAYSCALE)
test_img = cv2.imread(test_image_path)

if authorized_face_img is None or test_img is None:
    raise FileNotFoundError("Check that image paths are correct.")

# ---- Initialize ORB ----
orb = cv2.ORB_create()

# ---- Detect features on authorized face ----
kp1, des1 = orb.detectAndCompute(authorized_face_img, None)
if des1 is None:
    raise ValueError("No features detected in authorized face image.")

# Show authorized face with keypoints
auth_kp_img = cv2.drawKeypoints(authorized_face_img, kp1, None, color=(0, 255, 0))
plt.figure(figsize=(6, 5))
plt.title('Authorized Face Keypoints')
plt.imshow(cv2.cvtColor(auth_kp_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# ---- Convert test image to grayscale and detect faces ----
gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_test, 1.1, 4)

authorized = False

for (x, y, w, h) in faces:
    face_roi = gray_test[y:y+h, x:x+w]
    kp2, des2 = orb.detectAndCompute(face_roi, None)

    if des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        # Show top 10 matches
        match_img = cv2.drawMatches(authorized_face_img, kp1, face_roi, kp2, matches[:10], None, flags=2)
        plt.figure(figsize=(8, 6))
        plt.title('Top 10 Matches')
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # Check good matches
        good_matches = [m for m in matches if m.distance < 50]
        if len(good_matches) > 10:
            authorized = True

# ---- Output Result ----
print("Authorized" if authorized else "Unauthorized")
