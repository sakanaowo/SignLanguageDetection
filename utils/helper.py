import re
import matplotlib.pyplot as plt


def parse_training_log(log_text):
    """
    Phân tích log text và trả về dict chứa dữ liệu theo epoch.
    """
    epochs = []
    acc, val_acc = [], []
    loss, val_loss = [], []

    epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
    metrics_pattern = re.compile(
        r"categorical_accuracy: ([\d.]+) - loss: ([\d.]+) - val_categorical_accuracy: ([\d.]+) - val_loss: ([\d.]+)"
    )

    lines = log_text.strip().splitlines()
    for i in range(len(lines)):
        epoch_match = epoch_pattern.search(lines[i])
        if epoch_match:
            epoch = int(epoch_match.group(1))
            metric_match = metrics_pattern.search(lines[i + 1])  # Dòng tiếp theo chứa số liệu
            if metric_match:
                acc_val, loss_val, val_acc_val, val_loss_val = map(float, metric_match.groups())
                epochs.append(epoch)
                acc.append(acc_val)
                loss.append(loss_val)
                val_acc.append(val_acc_val)
                val_loss.append(val_loss_val)

    return {
        'epoch': epochs,
        'accuracy': acc,
        'val_accuracy': val_acc,
        'loss': loss,
        'val_loss': val_loss
    }


def plot_acc(log_dict):
    """
    Vẽ biểu đồ từ dữ liệu log.
    """
    epochs = log_dict['epoch']

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, log_dict['accuracy'], label='Train Accuracy')
    plt.plot(epochs, log_dict['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Accuracy')
    plt.legend()



def plot_loss(log_dict):
    epochs = log_dict['epoch']

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, log_dict['loss'], label='Train Loss')
    plt.plot(epochs, log_dict['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
