# Exno.4 – Program to Implement Object Labelling
import cv2
import numpy as np

def detect_and_label_objects(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found")
        return

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges (HSV) and their labels
    color_ranges = [
        ((0, 100, 100), (10, 255, 255), "Red"),
        ((25, 100, 100), (35, 255, 255), "Yellow"),
        ((100, 100, 100), (120, 255, 255), "Blue")
    ]

    for lower, upper, label in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

    cv2.imshow("Labeled Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run program
detect_and_label_objects("data/peacock.jpeg")
