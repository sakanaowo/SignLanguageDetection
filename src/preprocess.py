import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_config(config_path='src/config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_data(config_path='src/config.json', save=True):
    config = load_config(config_path)
    actions = config["ACTIONS"]
    sequence_length = config["SEQUENCE_LENGTH"]
    data_path = config["DATA_PATH"]
    processed_path = 'data/processed'

    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []

    for action in actions:
        # action_path = os.path.join(data_path, action)
        action_path = os.path.join(os.getcwd(), data_path, action)
        for sequence in np.array(os.listdir(action_path)).astype(int):
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_path, str(sequence), f"{frame_num}.npy")
                res = np.load(frame_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    if save:
        os.makedirs(processed_path, exist_ok=True)
        np.save(os.path.join(processed_path, 'X_train.npy'), X_train)
        np.save(os.path.join(processed_path, 'X_test.npy'), X_test)
        np.save(os.path.join(processed_path, 'y_train.npy'), y_train)
        np.save(os.path.join(processed_path, 'y_test.npy'), y_test)
        print(f"Saved preprocessed data to '{processed_path}'")

    return X_train, X_test, y_train, y_test
