import re
import matplotlib.pyplot as plt


def extract_log(log_text):
    log_data = {
        'epoch': [],
        'accuracy': [],
        'val_accuracy': [],
        'loss': [],
        'val_loss': []
    }

    # Tìm tất cả các epoch
    epoch_matches = re.finditer(r"Epoch (\d+)/\d+", log_text)
    for match in epoch_matches:
        epoch_num = int(match.group(1))
        log_data['epoch'].append(epoch_num)

    # Tìm các giá trị training accuracy và loss
    train_matches = re.findall(r"categorical_accuracy: ([0-9.]+) - loss: ([0-9.]+)", log_text)
    for acc, loss in train_matches:
        log_data['accuracy'].append(float(acc))
        log_data['loss'].append(float(loss))

    # Tìm các giá trị val_accuracy và val_loss
    val_matches = re.findall(r"val_categorical_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)", log_text)
    for val_acc, val_loss in val_matches:
        log_data['val_accuracy'].append(float(val_acc))
        log_data['val_loss'].append(float(val_loss))

    return log_data

def plot_loss(log_dict):
    """Vẽ biểu đồ Loss."""
    train_loss_epochs = range(len(log_dict['loss']))
    val_loss_epochs = range(len(log_dict['val_loss']))

    plt.figure(figsize=(12, 5))

    # Plot loss
    # plt.subplot(1, 2, 2)
    plt.plot(train_loss_epochs, log_dict['loss'], label='Train Loss', marker='o')
    plt.plot(val_loss_epochs, log_dict['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_acc(log_dict):
    """Vẽ biểu đồ Accuracy."""
    train_acc_epochs = range(len(log_dict['accuracy']))
    val_acc_epochs = range(len(log_dict['val_accuracy']))

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    # plt.subplot(1, 2, 1)
    plt.plot(train_acc_epochs, log_dict['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(val_acc_epochs, log_dict['val_accuracy'], label='Val Accuracy', marker='o')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
