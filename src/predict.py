import cv2
import numpy as np
import tensorflow as tf
from utils.draw import mediapipe_detection, draw_styled_landmarks
from src.keypoint_extraction import extract_keypoints
from src import config
import mediapipe as mp

# Load model
model = tf.keras.models.load_model('../models/action_model.h5')

# Load actions list from config
ACTIONS = config.ACTIONS
SEQUENCE_LENGTH = config.SEQUENCE_LENGTH

# For storing keypoints history
sequence = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-SEQUENCE_LENGTH:]

        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))

            if res[np.argmax(res)] > threshold:
                action = ACTIONS[np.argmax(res)]
                cv2.putText(image, action, (3, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow('Action Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
