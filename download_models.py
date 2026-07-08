"""
Downloads the pre-trained model files needed for age & gender detection.
Run this once before running detect_age_gender.py:

    python download_models.py
"""

import os
import urllib.request

os.makedirs("models", exist_ok=True)

# (urls, destination filename)
FILES = [
    # Face detector (OpenCV DNN, TensorFlow)
    ([
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    ], "models/opencv_face_detector.pbtxt"),
    ([
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb"
    ], "models/opencv_face_detector_uint8.pb"),

    # Age model
    ([
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt"
    ], "models/age_deploy.prototxt"),
    ([
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/age_net.caffemodel",
        "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1"
    ], "models/age_net.caffemodel"),

    # Gender model
    ([
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
    ], "models/gender_deploy.prototxt"),
    ([
        "https://github.com/spmallick/learnopencv/raw/master/AgeGender/gender_net.caffemodel",
        "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1"
    ], "models/gender_net.caffemodel"),
]


def download(urls, dest):
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return

    if not isinstance(urls, list):
        urls = [urls]

    last_error = None
    for url in urls:
        print(f"[downloading] {url} -> {dest}")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[done] {dest}")
            return
        except Exception as e:
            last_error = e
            print(f"[FAILED] {dest} from {url}: {e}")

    print(f"  -> Please download this file manually and place it in the 'models' folder. Last error: {last_error}")


if __name__ == "__main__":
    for urls, dest in FILES:
        download(urls, dest)

    print("\nAll downloads attempted. Check the 'models' folder.")
    print("If any file failed, see README.md for manual download links.")
