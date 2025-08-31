import cv2
import dlib
import numpy as np
from pathlib import Path

# Initialize dlib's face detector (HOG-based) and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(str(Path("resources/shape_predictor_68_face_landmarks.dat")))

def get_landmarks(image, detector, predictor):
    """Detect faces and return facial landmarks."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    landmarks = [predictor(gray, face) for face in faces]
    return faces, landmarks

def apply_face_swap(frame, faces, landmarks):
    """Placeholder for face swapping logic."""
    # For now, just draw rectangles around detected faces
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

def main():
    # Open video capture (webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect faces and landmarks
        faces, landmarks = get_landmarks(frame, face_detector, landmark_predictor)

        # Apply face swapping
        output_frame = apply_face_swap(frame, faces, landmarks)

        # Display the frame
        cv2.imshow("Live DeepFace Swap", output_frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
