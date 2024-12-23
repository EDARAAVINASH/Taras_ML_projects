import cv2
from playsound import playsound

# Load Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

eye_closed_frames = 0
drowsy_threshold = 15
alert_playing = False  # Flag to ensure the sound is played only once

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:
            eye_closed_frames += 1
        else:
            eye_closed_frames = 0

        if eye_closed_frames > drowsy_threshold:
            detected = True

    if detected:
        cv2.putText(frame, "DROWSINESS DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not alert_playing:
            playsound('alert.mp3', block=False)  # Play alert sound
            alert_playing = True
    else:
        cv2.putText(frame, "NOT DETECTED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        alert_playing = False

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
