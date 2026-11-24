# Real-Time Hand & Face AR Filter (Multi-Threaded) 🖐️😄🎥

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-0097A7?style=flat-square)

A high-FPS, real-time **Augmented Reality (AR) computer vision** application built with **Python**, **OpenCV**, and the **MediaPipe Tasks API**. It combines symmetric hand-gesture recognition, two-hand spatial detection, and facial blendshape expression tracking to trigger live camera filters and dynamic PNG emoji overlays.

---

### ✨ Key Features

- **Multi-Threaded Pipeline for Smooth 60 FPS:**
  - Dedicated background thread (`ThreadedCamera` with MJPG codec) for non-blocking 720p webcam capture.
  - Parallel AI inference (`ThreadPoolExecutor`) running `HandLandmarker` and `FaceLandmarker` concurrently, with pre-cached RGBA alpha-blending in RAM.
- **Symmetric Left & Right Hand Gesture Recognition:**
  - Single-hand overlays: `OK_HAND`, `THUMBS_UP`, `OPEN_PALM`, `PEACE`, and `ROCK_ON`.
  - Two-hand spatial gesture: Form a **Triangle** with both index fingers and thumbs to spawn a centered overlay.
- **Gesture-Controlled Camera Filters:**
  - **1 Finger (Index up):** Original camera feed
  - **2 Fingers (Ring + Pinky up):** Sepia tone filter
  - **3 Fingers (Index + Middle + Ring up):** Black & White (B&W) filter
- **Facial Blendshape Expression Tracking:**
  - Detects `SMILE`, `SHOCKED` (jaw open), and `WINK` in real time to render floating forehead overlays with frame-hold smoothing.

---

### 📂 Project Structure

```text
hand-tracking-filter/
├── main.py                  # Main application & parallel inference loop
├── requirements.txt         # Python dependencies
├── hand_landmarker.task     # MediaPipe Hand Landmarker model
├── face_landmarker.task     # MediaPipe Face Landmarker model
└── *.png                    # Transparent emoji overlay assets (9 files)
