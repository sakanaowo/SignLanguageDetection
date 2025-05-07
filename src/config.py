# # Dataset paths
# RAW_DATA_PATH = 'data/raw'
# PROCESSED_DATA_PATH = 'data/processed'
# MODEL_DIR = 'models'
#
# # Actions (classes) to recognize
# ACTIONS = ['hello', 'thanks', 'iloveyou']  # <-- bạn có thể cập nhật lại danh sách hành động này
#
# # Data collection parameters
# NO_SEQUENCES = 30
# SEQUENCE_LENGTH = 30
#
# # Model parameters
# EPOCHS = 200
# BATCH_SIZE = 16
# LEARNING_RATE = 0.0001
#
# # Input shape (33 pose + 468 face + 21 LH + 21 RH) * 3 = 1662
# NUM_KEYPOINTS = 1662
"""load config"""
import json
from pathlib import Path

config_path = Path(__file__).resolve().parent / 'config.json'
with open('config.json', 'r') as f:
    config = json.load(f)

ACTIONS = config['ACTIONS']
SEQUENCE_LENGTH = config['SEQUENCE_LENGTH']
NO_SEQUENCES = config['NO_SEQUENCES']
DATA_PATH = Path(__file__).parent / config['DATA_PATH']
X_TEST = Path(__file__).parent / config['X_TEST']
Y_TEST = Path(__file__).parent / config['Y_TEST']
