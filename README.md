# ActionDetection

---

Current Project Structure
```
📁 Action Detection/
├── data/
│   ├── raw/                ← dữ liệu .npy thu từ webcam
│   └── processed/          ← train/test split nếu cần
├── models/
│   └── action_model.h5     ← mô hình đã train
├── src/
│   ├── config.json         ← chứa cấu hình
│   ├── config.py           ← đọc từ config.json
│   ├── data_collection.py  ← thu thập dữ liệu từ webcam
│   ├── keypoint_extraction.py ← trích xuất pose/hand/face keypoints
│   ├── preprocess.py       ← xử lý dữ liệu thành X/y
│   ├── model.py            ← build, train và save model
│   └── predict.py          ← dự đoán hành động từ webcam
├── utils/
│   └── draw.py             ← MediaPipe + vẽ landmark
├── main.py                 ← chạy toàn bộ pipeline nếu cần
├── requirements.txt        ← thư viện cần thiết
└── README.md
```