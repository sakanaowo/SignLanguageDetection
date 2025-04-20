# ActionDetection
  
---  
## Project Objective
Build a human action recognition system from webcam using:
- MediaPipe to collect pose/hand/face landmarks
- Extract key points and train a gesture/action classification model
- Predict real-time actions from webcam
---
## Processing Process

| Step | Description                       | File                  |
|------|-----------------------------------|-----------------------|
| 1    | Collect data from webcam          | `src/data_collection` |
| 2    | Automatic save `.npy` key points  | `data/raw/ <action>`  |
| 3    | Autoupdate `ACTIONS`              | `src/config`          |
| 4    | Preprocess data into `X,y`        | `src/preprocess`      |
| 5    | Build and train model (`LSTM`)    | `src/model`           |
| 6    | Save model `.h5`                  | `models`              |
| 7    | Real-time prediction using webcam | `src/predict`         |

---
## Current Project Structure
```  
📁 Action Detection/
├── data/  
│   ├── raw/                    ← dữ liệu .npy thu từ webcam  
│   └── processed/              ← train/test split nếu cần  
├── models/  
│   └── action_model.h5         ← mô hình đã train  
├── src/  
│   ├── config.json             ← chứa cấu hình  
│   ├── config.py               ← đọc từ config.json  
│   ├── data_collection.py      ← thu thập dữ liệu từ webcam  
│   ├── keypoint_extraction.py  ← trích xuất pose/hand/face keypoints  
│   ├── preprocess.py           ← xử lý dữ liệu thành X/y  
│   ├── model.py                ← build, train và save model  
│   └── predict.py              ← dự đoán hành động từ webcam  
├── utils/  
│   └── draw.py                 ← MediaPipe + vẽ landmark  
├── build_n_train.py            ← chạy toàn bộ pipeline  
├── requirements.txt            ← thư viện cần thiết  
└── README.md  
```
---
## Completed Features
- Collect full key points from face/pose/hands using MediaPipe
- Extract key points into 1662-dimensional vectors
- Automatically add new actions and update configurations
- Process standard input data for training
- Build LSTM models and train
- Predict real-time actions on webcam
- Organize standard modular projects, easy to maintain, easy to upgrade
- Ready to deploy or improve
---
## Next Direction

---