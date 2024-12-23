import cv2

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cam = cv2.VideoCapture(0)

while True:
    success, frame = cam.read()
    
    if not success:
        print("Failed to capture image.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with face detection
    cv2.imshow("Face Detection", frame)

    # Break the loop when the 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
