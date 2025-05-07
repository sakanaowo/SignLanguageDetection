from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


# def build_model(input_shape, output_dim):
#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
#     model.add(LSTM(128, return_sequences=True, activation='relu'))
#     model.add(LSTM(64, return_sequences=False, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(output_dim, activation='softmax'))
#
#     model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#     return model

def build_model(input_shape, output_dim):
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=True, activation='tanh'), input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LayerNormalization())

    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(LayerNormalization())

    model.add(Bidirectional(LSTM(64, return_sequences=False, activation='tanh')))
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

    checkpoint = ModelCheckpoint(
        filepath="models/action_model.keras",
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, checkpoint]
    )
    # model.save("models/action_model.h5")
    print("Model trained and saved to models/action_model.keras")


def load_trained_model(model_path="models/action_model.keras"):
    """
    Load a trained Keras model from .h5 file.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        keras.Model: The loaded model.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from '{model_path}'")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
