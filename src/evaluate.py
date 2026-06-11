import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model

from preprocessing import load_and_preprocess_data


def evaluate():

    _, x_test, _, y_test = (
        load_and_preprocess_data()
    )

    model = load_model(
        "../models/cnn_model.keras"
    )

    predictions = model.predict(x_test)

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(
        classification_report(
            y_true,
            y_pred,
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
