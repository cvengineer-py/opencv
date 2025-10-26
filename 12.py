import cv2 as cv
import matplotlib.pyplot as plt

# Load pre-trained pose model
net = cv.dnn.readNetFromTensorflow("data/graph_opt.pb")

# Load input image
frame = cv.imread("data/human.jpeg")
h, w = frame.shape[:2]

# Prepare input blob
blob = cv.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
net.setInput(blob)
out = net.forward()

# Body parts and connections
BODY_PARTS = {
    "Nose": 0, "Neck": 1,
    "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7,
    "RHip": 8, "RKnee": 9, "RAnkle": 10,
    "LHip": 11, "LKnee": 12, "LAnkle": 13
}

POSE_PAIRS = [
    ("Neck", "RShoulder"), ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("Neck", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("Neck", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
]

# Detect keypoints
points = []
for i in range(len(BODY_PARTS)):
    heatMap = out[0, i, :, :]
    _, conf, _, point = cv.minMaxLoc(heatMap)
    x = int(w * point[0] / out.shape[3])
    y = int(h * point[1] / out.shape[2])
    points.append((x, y) if conf > 0.2 else None)

# Draw skeleton
for pair in POSE_PAIRS:
    partA, partB = pair
    idA, idB = BODY_PARTS[partA], BODY_PARTS[partB]
    if points[idA] and points[idB]:
        cv.line(frame, points[idA], points[idB], (0, 255, 0), 3)
        cv.circle(frame, points[idA], 3, (0, 0, 255), -1)
        cv.circle(frame, points[idB], 3, (0, 0, 255), -1)

# Show result
plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Pose Estimation")
plt.show()
