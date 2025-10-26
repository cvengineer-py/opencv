# Exno.6 License Plate Identification (Minimal)
import cv2
import numpy as np
import pytesseract
import os

# --- Paths ---
pytesseract.pytesseract.tesseract_cmd = r"D:\Softwares\tessaract\tesseract.exe"
cascade_path = r"data\haarcascade_russian_plate_number.xml"

if not os.path.exists(cascade_path):
    raise FileNotFoundError("Haarcascade XML file not found!")

# Load Cascade
plate_cascade = cv2.CascadeClassifier(cascade_path)

# Minimal State Dictionary
states = {
    "TN": "Tamil Nadu"
}

def extract_number_plate(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Image not found.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray)

    for (x, y, w, h) in plates:
        plate = img[y:y+h, x:x+w]
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate_bin = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        # OCR using Tesseract
        text = pytesseract.image_to_string(plate_bin, config='--psm 8')
        number = ''.join(e for e in text if e.isalnum()).upper()
        state_code = number[:2] 
        state_name = states.get(state_code, "Unknown")

        print(f"\nDetected Number: {number}")
        print(f"State: {state_name}")

        # Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(img, number, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(img, state_name, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Save & Display
        cv2.imwrite("Detected_Plate.png", plate)
        cv2.imwrite("Detected_Image.png", img)
        cv2.imshow("Detected Plate", plate)
        cv2.imshow("Full Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
extract_number_plate(r"data\car.jpg")
