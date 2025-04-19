from src.preprocess import preprocess_data
from src.model import build_model, train_model


def main():
    # 1. Tiền xử lý dữ liệu: X, y từ raw → processed
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()

    # 2. Build model
    print("Building model...")
    model = build_model(
        input_shape=X_train.shape[1:],  # (30, 1662)
        output_dim=y_train.shape[1]  # số lượng class
    )

    # 3. Train model
    print("Training model...")
    train_model(model, X_train, y_train, X_test, y_test)

    print("Done!")


if __name__ == '__main__':
    main()
