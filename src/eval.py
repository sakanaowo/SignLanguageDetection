from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import config
from src.model import load_trained_model


def evaluate_model(model, X_test, y_test, class_names=None):
    # Dự đoán xác suất
    y_pred_probs = model.predict(X_test)
    # Lấy nhãn dự đoán và nhãn thật
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Xác định số lớp thực tế
    labels = sorted(np.unique(y_true))
    # Cập nhật class_names nếu được cung cấp
    if class_names is not None:
        class_names = [class_names[i] for i in labels]

    # Báo cáo phân loại
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names))

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else labels,
                yticklabels=class_names if class_names else labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = load_trained_model()
    class_names = config['ACTIONS']
    X_TEST = np.load(config['X_TEST'])
    Y_TEST = np.load(config['Y_TEST'])
    evaluate_model(model, X_TEST, Y_TEST, class_names)
