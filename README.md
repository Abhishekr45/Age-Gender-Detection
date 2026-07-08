# Real-Time Age & Gender Detection

This project detects a person's gender and age range from a live camera feed or an uploaded image using pre-trained deep learning models and OpenCV.

It includes two modes:

1. Desktop mode - opens a local camera window and shows predictions in real time.
2. Web mode - runs a Flask web app where the browser captures frames and sends them to the backend for prediction.

---

## What this project does

The system works in three steps:

1. Face detection
   - A pre-trained OpenCV DNN face detector finds faces in the image.

2. Gender prediction
   - A Caffe-based gender model predicts whether the face is male or female.

3. Age prediction
   - A Caffe-based age model predicts an age range such as (25-32).

The app then draws bounding boxes around detected faces and displays the predicted gender and age on the screen.

---

## Project structure

```text
age_gender_detection/
├── detect_age_gender.py         # Desktop version
├── download_models.py           # Downloads the required model files
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── models/                     # Downloaded pretrained models
└── webapp/
    ├── app.py                  # Flask backend
    └── templates/
        └── index.html          # Web frontend UI
```

---

## Requirements

Before running the project, make sure you have:

- Python 3.10 or newer
- A webcam connected (for live camera mode)
- Internet access to download the model files

This project has been tested with:

- Python 3.12
- OpenCV 4.8.0.76
- NumPy 1.26.x
- Flask 3.x

---

## Setup on Windows (PowerShell)

Open PowerShell in the project folder and run the following commands:

```powershell
cd D:\DATA_SCIENCE\DeepLearning\age_gender_detection
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python download_models.py
```

If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## How to run the desktop version

Run:

```powershell
python detect_age_gender.py
```

What happens:

- The camera opens in a local OpenCV window.
- Detected faces are highlighted with boxes.
- Labels show the predicted gender and age range.
- Press q to close the window.

Optional arguments:

```powershell
python detect_age_gender.py --camera 1
python detect_age_gender.py --image photo.jpg
```

---

## How to run the web app

1. Start the backend server:

```powershell
cd webapp
python app.py
```

2. Open your browser and visit:

```text
http://127.0.0.1:5000
```

3. Allow camera access when the browser asks.

4. Click Start camera to begin live detection.

The web app sends video frames from the browser to the Flask backend, where the model runs and returns predictions. The result is drawn back on the page in real time.

---

## How the project works internally

### Backend

The Flask server in webapp/app.py:

- loads the face detector and the age/gender model files,
- receives image frames from the browser,
- runs face detection and prediction,
- returns JSON results back to the frontend.

### Frontend

The HTML/JavaScript UI in webapp/templates/index.html:

- captures video from the webcam,
- draws the video stream to a canvas,
- sends frames to the backend,
- displays bounding boxes and prediction labels on the page.

### Model files

The project uses:

- face detector model files
- age model files
- gender model files

These are downloaded by download_models.py into the models folder.

---

## Troubleshooting

### 1. Activation command fails

If you see an error like source is not recognized, use:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Missing model files error

If the app says the model files are missing, run:

```powershell
python download_models.py
```

### 3. OpenCV or NumPy errors

If you get errors related to cv2 or NumPy, make sure the environment matches the project requirements:

```powershell
python -m pip install --force-reinstall "opencv-python==4.8.0.76"
python -m pip install --force-reinstall "numpy<2"
```

### 4. Camera not working

- Make sure your webcam is connected.
- Allow camera access in the browser.
- Try another camera index if needed:

```powershell
python detect_age_gender.py --camera 1
```

---

## Notes and limitations

- The age model predicts an age range, not an exact age.
- Accuracy depends on lighting, face angle, and image quality.
- Good lighting and a straight-facing face improve results.
- This project runs locally and does not require a GPU.

---

## Summary

You can use this project in two ways:

- Desktop app for quick local testing.
- Web app for a browser-based experience.

If you want, you can later extend it to:

- upload images instead of using the webcam,
- show confidence scores more clearly,
- deploy it to a cloud platform.
