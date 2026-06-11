from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)

from preprocessing import load_and_preprocess_data
from model import build_model


MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(exist_ok=True)


def train():

    x_train, x_test, y_train, y_test = (
        load_and_preprocess_data()
    )

    model = build_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        ModelCheckpoint(
            filepath=MODEL_DIR / "cnn_model.keras",
            save_best_only=True,
            monitor="val_accuracy",
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )

    print(f"Test Accuracy: {test_acc:.4f}")

    return history


if __name__ == "__main__":
    train()
