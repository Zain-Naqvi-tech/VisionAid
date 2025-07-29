import cv2 as cv          # For capturing video and drawing boxes
import numpy as np     # For distance calculations
from ultralytics import YOLO  # For object detection (YOLOv8) (Will be used for both hands and objects)
import pyttsx3 # For text to speech
import threading # For running speech in a separate thread (so it doesn't block video)
import time # For tracking cooldowns between speech events

# Initialize text-to-speech engine
engine = pyttsx3.init()
# Set the rate (speed) of speech (words per minute)
engine.setProperty('rate', 150)
# Set the volume of the speech (0.0 to 1.0)
engine.setProperty('volume', 1.0)

engine.say('Hello, Welcome to the Object Detection System')
engine.runAndWait() # This is used to speak the text

# Define a function to speak text in a non-blocking way
# This is important because engine.runAndWait() is blocking (it waits for speech to finish)
# By running it in a separate thread, the main video loop can keep running while speech happens
# This prevents the video from freezing or lagging when speaking
# The function takes a string (text) and speaks it asynchronously
def speak(text):
    t = threading.Thread(target=lambda: engine.say(text) or engine.runAndWait())
    t.daemon = True
    t.start()

model = YOLO("yolov8n-pose.pt") # Used to detect human features
model2 = YOLO("yolov8s.pt") # Used to detect objects (we will need to filter out the person class)
cap = cv.VideoCapture(0) # Used to capture video from the camera

# Track last spoken object and cooldown for each hand
last_spoken_object_L = None
last_spoken_time_L = 0
last_spoken_object_R = None
last_spoken_time_R = 0
cooldown_seconds = 2  # Minimum seconds between speech events

# Main loop to continuously capture frames from the camera
while True:
    # Reset closest object variables for this frame
    closest_object_L = None 
    closest_distance_L = float('inf') #large number
    closest_object_R = None 
    closest_distance_R = float('inf') #large number

    ret, img = cap.read() #read the frame from the camera
    
    results = model(img) #detect the human features
    results2 = model2(img) #detect the objects

    # For each detected person, extract hand positions
    for person in results[0].keypoints.xy:
        # Extract the coordinates for the left and right wrists (hands)
        # Index 9: left wrist, Index 10: right wrist 
        left_hand = person[9]
        right_hand = person[10]

        # Draw a black circle on the left hand (wrist)
        cv.circle(img, (int(left_hand[0]), int(left_hand[1])), 10, (0,0,0), -1)
        # Draw a black circle on the right hand (wrist)
        cv.circle(img, (int(right_hand[0]), int(right_hand[1])), 10, (0,0,0), -1)

        # For each detected object, check distance to both hands
        for box in results2[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Bounding box coordinates
            label = model2.names[int(box.cls)] # Object label
            conf = box.conf[0] #Detection confidence

            if label == 'person' or label == 'chair' or label == 'laptop' or label == 'dining table': #Some objects that can be ignored
                continue

            # Calculate object center
            object_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Calculate Euclidean distance from object center to each hand
            left_dist = np.linalg.norm(np.array(object_center) - np.array(left_hand))
            right_dist = np.linalg.norm(np.array(object_center) - np.array(right_hand))

            # Update closest object for left hand if this one is closer
            if left_dist < closest_distance_L:
                closest_distance_L = left_dist
                closest_object_L = (label, (x1, y1, x2, y2))
            # Update closest object for right hand if this one is closer
            if right_dist < closest_distance_R:
                closest_distance_R = right_dist
                closest_object_R = (label, (x1, y1, x2, y2))

            cv.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2) #draw a rectangle around the object
            cv.putText(img, f"{label} {conf:.2f}", (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        current_time = time.time() # Get the current time (for cooldown logic)
        if closest_object_L:
            label, (x1, y1, x2, y2) = closest_object_L
            cv.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv.putText(img, f"Closest to Left: {label}", (x1, y1-30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            if (last_spoken_object_L != label) and (current_time - last_spoken_time_L > cooldown_seconds):
                # Use the speak() function to say the label in a separate thread (non-blocking)
                speak(f"Closest to Left: {label}")
                # Update the last spoken object and time for the left hand
                last_spoken_object_L = label
                last_spoken_time_L = current_time
        if closest_object_R:
            label, (x1, y1, x2, y2) = closest_object_R
            cv.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2) #draw a rectangle around the object
            cv.putText(img, f"Closest to Right: {label}", (x1, y1-50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)


            if (last_spoken_object_R != label) and (current_time - last_spoken_time_R > cooldown_seconds):
                speak(f"Closest to Right: {label}")
                last_spoken_object_R = label
                last_spoken_time_R = current_time

    cv.imshow('Hands Highlighted', img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
