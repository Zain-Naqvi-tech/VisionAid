
This Python-based project detects how close your hand is to objects using real-time webcam input. It uses YOLOv8 pose detection to find hand landmarks, calculates the distance to detected objects, and gives real-time audio feedback when your hand is close.

Features:
- YOLOv8 (via Ultralytics)

- OpenCV

- NumPy

- pyttsx3 (Text-to-Speech)

- Real-time webcam feed

- Calculates distance between hand and nearby object

- Speaks out the closest distance

How to Run:

    1. Clone the repository

    2. Install dependencies:

        pip install -r requirements.txt
    
    3. Run:
    
        python main.py