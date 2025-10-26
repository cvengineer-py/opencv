import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# ----- Paths -----
MODEL_PATH = r'data/action_recognition_model.h5'
TEST_IMAGE = r'data//run.jpg'
action_classes = ['walking', 'running', 'jumping', 'standing', 'sitting', 'falling', 'other']

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Determine expected input size & channels
_, H, W, C = model.input_shape

# Load and preprocess image
img = cv2.imread(TEST_IMAGE)
if C == 1:
    img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_proc = cv2.resize(img_proc, (W, H))[:, :, np.newaxis]
else:
    img_proc = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W, H))

x = np.expand_dims(img_proc.astype('float32') / 255.0, axis=0)

# Predict action
preds = model.predict(x)[0]
top_label = action_classes[np.argmax(preds)]

# Display results
print(f"Predicted Action: {top_label} ({np.max(preds)*100:.1f}%)")
cv2.putText(img, top_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
