import cv2 as cv
import matplotlib.pyplot as plt

# ---- Load pre-trained TensorFlow pose model ----
net = cv.dnn.readNetFromTensorflow(r"data\graph_opt.pb")

# ---- Parameters ----
inWidth = 368
inHeight = 368
thr = 0.2  # confidence threshold

# ---- Body parts and skeleton connections ----
BODY_PARTS = {
    "Nose": 0, "Neck": 1,
    "Right Shoulder": 2, "Right Elbow": 3, "Right Wrist": 4,
    "Left Shoulder": 5, "Left Elbow": 6, "Left Wrist": 7,
    "Right Hip": 8, "Right Knee": 9, "Right Ankle": 10,
    "Left Hip": 11, "Left Knee": 12, "Left Ankle": 13,
    "Right Eye": 14, "Left Eye": 15,
    "Right Ear": 16, "Left Ear": 17
}

POSE_PAIRS = [
    ("Neck", "Right Shoulder"), ("Neck", "Left Shoulder"),
    ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"),
    ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
    ("Neck", "Right Hip"), ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
    ("Neck", "Left Hip"), ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
    ("Neck", "Nose"),
    ("Nose", "Right Eye"), ("Right Eye", "Right Ear"),
    ("Nose", "Left Eye"), ("Left Eye", "Left Ear")
]

# ---- Load image ----
frame = cv.imread(r"data\human.jpeg")
frameHeight, frameWidth = frame.shape[:2]

# ---- Prepare input for network ----
blob = cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                            (127.5, 127.5, 127.5), swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()[:, :len(BODY_PARTS), :, :]

# ---- Detect keypoints ----
points = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = int((frameWidth * point[0]) / out.shape[3])
    y = int((frameHeight * point[1]) / out.shape[2])
    points.append((x, y) if conf > thr else None)

# ---- Draw skeleton ----
for pair in POSE_PAIRS:
    partFrom, partTo = pair
    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]
    if points[idFrom] and points[idTo]:
        cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
        cv.circle(frame, points[idFrom], 3, (0, 0, 255), -1)
        cv.circle(frame, points[idTo], 3, (0, 0, 255), -1)

# ---- Display result ----
plt.figure(figsize=(10, 6))
plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Human Pose Estimation")
plt.show()
