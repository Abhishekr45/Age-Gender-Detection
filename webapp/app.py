"""
Flask web app for real-time Age & Gender Detection.

The browser captures frames from your webcam (via JavaScript) and sends
them to this backend, which runs the deep learning models and returns
the detected face boxes + age/gender labels as JSON. The frontend then
draws the results on top of the video.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import base64

try:
    import cv2
    import numpy as np
    from flask import Flask, render_template, request, jsonify
except ImportError as exc:
    raise SystemExit(
        "Missing required packages. Install them with:\n"
        "  python -m pip install -r requirements.txt\n"
        f"Original error: {exc}"
    ) from exc

app = Flask(__name__)

# ---------------------------------------------------------
# Model paths (same models used by the desktop script)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

FACE_PROTO = os.path.join(MODELS_DIR, "opencv_face_detector.pbtxt")
FACE_MODEL = os.path.join(MODELS_DIR, "opencv_face_detector_uint8.pb")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")
GENDER_PROTO = os.path.join(MODELS_DIR, "gender_deploy.prototxt")
GENDER_MODEL = os.path.join(MODELS_DIR, "gender_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
PADDING = 20

face_net = age_net = gender_net = None


def load_models():
    global face_net, age_net, gender_net
    required = [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL, GENDER_PROTO, GENDER_MODEL]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing model files:\n" + "\n".join(missing) +
            "\n\nRun 'python download_models.py' from the project root first."
        )
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("Models loaded successfully.")


load_models()


def detect_faces(frame, conf_threshold=0.7):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123],
                                  swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
    return boxes


def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    gender_conf = float(gender_preds[0].max())

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    age_conf = float(age_preds[0].max())

    return gender, gender_conf, age, age_conf


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image sent from the browser canvas
    img_data = data["image"].split(",")[-1]
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    results = []
    boxes = detect_faces(frame)
    for (x1, y1, x2, y2) in boxes:
        fy1, fy2 = max(0, y1 - PADDING), min(frame.shape[0] - 1, y2 + PADDING)
        fx1, fx2 = max(0, x1 - PADDING), min(frame.shape[1] - 1, x2 + PADDING)
        face_img = frame[fy1:fy2, fx1:fx2]
        if face_img.size == 0:
            continue
        gender, gender_conf, age, age_conf = predict_age_gender(face_img)
        results.append({
            "box": [x1, y1, x2, y2],
            "gender": gender,
            "gender_confidence": round(gender_conf * 100, 1),
            "age": age,
            "age_confidence": round(age_conf * 100, 1),
        })

    return jsonify({"faces": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
