import cv2
import numpy as np

# ---- Paths ----
weights = r"data\yolov4.weights"
cfg     = r"data\yolov4.cfg"
names   = r"data\coco.names"
video   = r"data\traffic.mp4"
# ---- Load YOLO ----
net = cv2.dnn.readNet(weights, cfg)
layers = [net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
classes = [line.strip() for line in open(names)]

cap = cv2.VideoCapture(video)
count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255, (608,608), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layers)

    boxes, confs, ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = np.argmax(scores)
            conf = scores[cid]
            if conf > 0.3 and classes[cid] in ["car","truck","bus","motorbike"]:
                cx, cy = int(det[0]*w), int(det[1]*h)
                bw, bh = int(det[2]*w), int(det[3]*h)
                x, y = max(0,cx-bw//2), max(0,cy-bh//2)
                boxes.append([x,y,bw,bh]); confs.append(float(conf)); ids.append(cid)

    for i in cv2.dnn.NMSBoxes(boxes, confs, 0.3, 0.4).flatten():
        x,y,bw,bh = boxes[i]
        cv2.rectangle(frame,(x,y),(x+bw,y+bh),(0,255,0),2)
        cv2.putText(frame, classes[ids[i]], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,0),2)
        if y > h//2: count += 1

    cv2.putText(frame,f'Count: {count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Traffic", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
