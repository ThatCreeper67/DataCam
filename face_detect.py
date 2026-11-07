"""
Real-time Face Detection with OpenCV DNN (SSD Caffe Model)
Detects faces from webcam and draws bounding boxes (hitboxes) around them.
Press 'q' to quit.
"""
import cv2
import numpy as np
import os
import sys

# Paths to model files
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
PROTO_PATH = os.path.join(MODEL_DIR, "deploy.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.5

def load_model():
    """Load the Caffe face detection model."""
    if not os.path.exists(PROTO_PATH):
        sys.exit(f"Error: {PROTO_PATH} not found")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: {MODEL_PATH} not found")

    # OpenCV 5.x uses readNet for all formats
    net = cv2.dnn.readNet(MODEL_PATH, PROTO_PATH)
    # Try to use CUDA if available
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Using CUDA acceleration")
    else:
        print("Using CPU (CUDA not available)")
    return net

def detect_faces(net, frame):
    """Run face detection on a frame, return list of (x1, y1, x2, y2, confidence)."""
    h, w = frame.shape[:2]

    # Create blob from image (resize to 300x300, normalize)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > CONFIDENCE_THRESHOLD:
            # Scale bounding box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = (int(value) for value in box.astype("int"))

            # Ensure coordinates are within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            faces.append((x1, y1, x2, y2, confidence))

    return faces

def draw_hitboxes(frame, faces):
    """Draw bounding boxes and confidence labels on the frame."""
    for (x1, y1, x2, y2, conf) in faces:
        # Draw the bounding box (hitbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence label
        label = f"Face: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Draw corner markers for "hitbox" effect
        corner_len = 20
        color = (0, 255, 0)
        thickness = 3
        # Top-left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Top-right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    return frame

def main():
    print("Loading face detection model...")
    net = load_model()

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        sys.exit("Error: Could not open webcam. Try changing the camera index (0, 1, 2...).")

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting face detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame")
            break

        # Mirror for selfie-view
        frame = cv2.flip(frame, 1)

        # Detect faces
        faces = detect_faces(net, frame)

        # Draw hitboxes
        frame = draw_hitboxes(frame, faces)

        # Show face count
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Face Detection - Hitboxes", frame)

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()