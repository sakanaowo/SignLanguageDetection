from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    model.save("models/action_model.h5")
    print("Model trained and saved to models/action_model.h5")
