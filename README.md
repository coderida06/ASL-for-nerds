# 🤟 ASL for Nerds

An interactive American Sign Language (ASL) learning application built with Python, OpenCV, MediaPipe, and CustomTkinter.

## Features

* Real-time hand tracking using MediaPipe
* ASL alphabet recognition (A-Z)
* Reference images for each sign
* Instant feedback on detected signs
* Modern dark-themed interface
* Webcam-based learning experience
* Special motion detection for the letter J

## Technologies Used

* Python
* CustomTkinter
* OpenCV
* MediaPipe
* NumPy
* Pillow

## Project Structure

```text
ASL-for-Nerds/
│
├── main.py
├── images/
│   ├── A.jpg
│   ├── B.jpg
│   ├── ...
│   └── Z.jpg
├── requirements.txt
└── README.md
```

## How It Works

1. Select an ASL letter.
2. View the reference image.
3. Show the sign to your webcam.
4. The application analyzes your hand landmarks.
5. Receive instant feedback when the correct sign is detected.

## Notes

* Good lighting improves recognition accuracy.
* Keep your hand clearly visible inside the camera frame.
* Ensure your webcam is connected and accessible.

## Author

Built by Rida as a computer vision and accessibility learning project.

Will be working the 2.0 version of this that teaches more than just the alphabets.
