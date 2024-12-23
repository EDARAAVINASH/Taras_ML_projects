import cv2
import os
import numpy as np
import pyttsx3  # For text-to-speech functionality

def load_templates():
    """
    Load currency template images and compute their ORB descriptors.
    """
    # Path to the folder containing currency templates
    template_path = "templates/"  

    # List of currency denominations to load
    denominations = [10, 20, 50, 100, 200, 500, 2000]  # Modify as needed

    # ORB detector
    orb = cv2.ORB_create()

    # Dictionary to store templates
    templates = {}

    print("Looking for templates in:", os.path.abspath(template_path))  # Debugging path

    for denom in denominations:
        # Check for .jpg and .jpeg extensions
        jpg_file = os.path.join(template_path, f"{denom}.jpg")
        jpeg_file = os.path.join(template_path, f"{denom}.jpeg")
        
        # Select the correct file path
        if os.path.exists(jpg_file):
            image_file = jpg_file
        elif os.path.exists(jpeg_file):
            image_file = jpeg_file
        else:
            print(f"Error: Template for {denom} INR not found (neither .jpg nor .jpeg).")
            continue

        # Load the template image in grayscale
        template_image = cv2.imread(image_file, 0)
        
        if template_image is not None:
            # Compute ORB keypoints and descriptors
            keypoints, descriptors = orb.detectAndCompute(template_image, None)
            templates[denom] = (keypoints, descriptors, template_image)
            print(f"Loaded template for {denom} INR.")
        else:
            print(f"Error: Unable to read image for {denom} INR at {image_file}.")
    
    return templates

def match_currency(frame, templates, orb, bf, min_good_matches=10):
    """
    Match the currency note in the frame with templates using ORB and BFMatcher.
    Parameters:
        - frame: The video frame to process.
        - templates: Loaded currency templates.
        - orb: The ORB detector object.
        - bf: The BFMatcher object.
        - min_good_matches: Minimum number of good matches to consider a valid match.
    Returns:
        - best_match: The detected currency denomination, or None if no match is found.
        - max_good_matches: The number of good matches for the best match found.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the frame
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    best_match = None
    max_good_matches = 0

    for denom, (template_keypoints, template_descriptors, template_image) in templates.items():
        # Match descriptors using BFMatcher
        if frame_descriptors is not None and template_descriptors is not None:
            matches = bf.knnMatch(template_descriptors, frame_descriptors, k=2)

            # Apply the ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # Ratio test
                    good_matches.append(m)

            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match = denom

    # If the best match has fewer good matches than the minimum threshold, return None
    if max_good_matches < min_good_matches:
        best_match = None

    return best_match, max_good_matches

def speak(text):
    """
    Use text-to-speech to announce the detected note.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Set speech speed
    engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)
    engine.say(text)
    engine.runAndWait()

def main():
    # Load templates
    templates = load_templates()
    if not templates:
        print("No templates loaded. Please check the templates folder and file names.")
        return

    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Starting real-time currency detection. Press 'q' to exit.")

    last_spoken = None  # To prevent repeating the same voice note

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture video.")
            break

        # Match the currency in the frame
        best_match, max_good_matches = match_currency(frame, templates, orb, bf)

        # Display results
        if best_match:
            result_text = f"Best Match: {best_match} INR"
            if last_spoken != best_match:
                speak(f"{best_match} rupees note detected")
                last_spoken = best_match
        else:
            result_text = "No Match Found"

        # Display the result on the frame
        cv2.putText(frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with result
        cv2.imshow("Currency Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
