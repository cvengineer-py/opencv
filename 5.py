# Exno.5 – Face Detection using DNN
import cv2

def face_detection_dnn(image_path):
    # Load model
    net = cv2.dnn.readNetFromCaffe("data/deploy.prototxt",
                                   "data/res10_300x300_ssd_iter_140000.caffemodel")

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 1.0, (300, 300))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        if detections[0, 0, i, 2] > 0.5:
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
face_detection_dnn("data/face.jpeg")
