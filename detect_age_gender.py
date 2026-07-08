"""
Real-Time Age & Gender Detection using Deep Learning (OpenCV DNN)
------------------------------------------------------------------
This script opens your webcam, detects faces in real time, and
predicts the AGE RANGE and GENDER of each detected face using
pre-trained deep learning models (Caffe).

Author: Generated for the user's DL project
"""

import cv2
import numpy as np
import argparse
import os

# ---------------------------------------------------------
# 1. Paths to the pre-trained model files
#    (Download instructions are in README.md / download_models.py)
# ---------------------------------------------------------
FACE_PROTO = "models/opencv_face_detector.pbtxt"
FACE_MODEL = "models/opencv_face_detector_uint8.pb"

AGE_PROTO = "models/age_deploy.prototxt"
AGE_MODEL = "models/age_net.caffemodel"

GENDER_PROTO = "models/gender_deploy.prototxt"
GENDER_MODEL = "models/gender_net.caffemodel"

# Mean values used during training of the age/gender models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

PADDING = 20  # extra pixels around detected face box


def check_models_exist():
    """Verify all required model files are present before running."""
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("ERROR: The following model files are missing:")
        for m in missing:
            print("  -", m)
        print("\nPlease run 'python download_models.py' first, "
              "or see README.md for manual download links.")
        exit(1)


def load_models():
    """Load face detector, age model, and gender model into OpenCV DNN."""
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    return face_net, age_net, gender_net


def detect_faces(face_net, frame, conf_threshold=0.7):
    """
    Runs the face detector on a frame and returns a list of
    bounding boxes [(x1, y1, x2, y2), ...] for detected faces.
    """
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  [104, 117, 123], swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append((x1, y1, x2, y2))
    return boxes


def predict_age_gender(face_img, age_net, gender_net):
    """Given a cropped face image, predict gender and age bucket."""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                  MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_confidence = gender_preds[0].max()

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    age_confidence = age_preds[0].max()

    return gender, gender_confidence, age, age_confidence


def run_webcam(camera_index=0):
    check_models_exist()
    face_net, age_net, gender_net = load_models()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam. Try a different --camera index.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        result_frame = frame.copy()
        boxes = detect_faces(face_net, frame)

        if not boxes:
            cv2.putText(result_frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for (x1, y1, x2, y2) in boxes:
            # Add padding around the face box, staying within frame bounds
            fy1 = max(0, y1 - PADDING)
            fy2 = min(frame.shape[0] - 1, y2 + PADDING)
            fx1 = max(0, x1 - PADDING)
            fx2 = min(frame.shape[1] - 1, x2 + PADDING)

            face_img = frame[fy1:fy2, fx1:fx2]
            if face_img.size == 0:
                continue

            gender, gender_conf, age, age_conf = predict_age_gender(
                face_img, age_net, gender_net)

            label = f"{gender} ({gender_conf*100:.0f}%), {age} ({age_conf*100:.0f}%)"

            cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(result_frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow("Age & Gender Detection - press 'q' to quit", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_on_image(image_path):
    """Optional: run detection on a single saved image instead of webcam."""
    check_models_exist()
    face_net, age_net, gender_net = load_models()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not read image at {image_path}")
        return

    boxes = detect_faces(face_net, frame)
    for (x1, y1, x2, y2) in boxes:
        fy1, fy2 = max(0, y1 - PADDING), min(frame.shape[0] - 1, y2 + PADDING)
        fx1, fx2 = max(0, x1 - PADDING), min(frame.shape[1] - 1, x2 + PADDING)
        face_img = frame[fy1:fy2, fx1:fx2]
        if face_img.size == 0:
            continue
        gender, gender_conf, age, age_conf = predict_age_gender(face_img, age_net, gender_net)
        label = f"{gender} ({gender_conf*100:.0f}%), {age} ({age_conf*100:.0f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path = "output_result.jpg"
    cv2.imwrite(out_path, frame)
    print(f"Saved result to {out_path}")
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Age & Gender Detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--image", type=str, default=None,
                         help="Path to a single image instead of using the webcam")
    args = parser.parse_args()

    if args.image:
        run_on_image(args.image)
    else:
        run_webcam(args.camera)
