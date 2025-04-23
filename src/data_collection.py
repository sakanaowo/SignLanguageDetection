import os
import numpy as np
import cv2
import json
import mediapipe as mp
from utils.draw import mediapipe_detection, draw_styled_landmarks
from src.keypoint_extraction import extract_keypoints
from src import config

mp_holistic = mp.solutions.holistic


def add_new_action(action_name, config_path='config.json'):
    with open(config_path, 'r+') as f:
        config_data = json.load(f)
        if action_name in config_data['ACTIONS']:
            print(f"'{action_name}' already exists in ACTIONS.")
            return False
        config_data['ACTIONS'].append(action_name)
        f.seek(0)
        json.dump(config_data, f, indent=4)
        f.truncate()
    print(f"'{action_name}' has been added to config.json ACTIONS list.")
    return True


def collect_data(actions, no_sequences=config.NO_SEQUENCES, sequence_length=config.SEQUENCE_LENGTH,
                 data_path=config.DATA_PATH):
    print("Collecting data into:", config.DATA_PATH)
    for action in actions:
        for sequence in range(no_sequences):
            os.makedirs(os.path.join(data_path, action, str(sequence)), exist_ok=True)

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):

                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    np.save(os.path.join(data_path, action, str(sequence), str(frame_num)), keypoints)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

    cap.release()
    cv2.destroyAllWindows()
