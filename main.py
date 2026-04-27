import cv2
import numpy as np
import tensorflow as tf

print("Starting ASL Detection...")

#  Load model
model = tf.keras.models.load_model("asl_model.h5")
print("Model Loaded")
print(model.summary())

#  Class labels (A-Z)
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

IMG_SIZE = 160

#  Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not accessible")
    exit()

print("Camera Started")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    #  Flip for mirror view
    frame = cv2.flip(frame, 1)

    #  ROI (Region of Interest)
    x1, y1, x2, y2 = 100, 100, 400, 400
    roi = frame[y1:y2, x1:x2]

    #  Preprocess image
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    #  NEW: Hand presence check
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    #  If no hand detected
    if np.std(gray) < 10:
        label = "No Hand"
        confidence = 0

    else:
        #  Prediction
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        #  Confidence filtering
        if confidence > 0.90:
            label = classes[class_index]
        else:
            label = "Uncertain"

    #  Display result
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    #  Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    #  Show output
    cv2.imshow("ASL Detection", frame)

    #  Exit with 'q' or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()