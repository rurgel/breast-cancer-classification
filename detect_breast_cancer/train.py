from processing.data_manager import load_dataset
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import Model, layers, metrics, losses
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

from detect_breast_cancer.config.core import config, MODEL_PATH, BASE_PATH


def model():
    enb3 = EfficientNetB3(include_top=False, input_shape=(
        config.CROP, config.CROP, 3), classes=2)

    input = layers.Input(shape=config.IMAGE_SIZE + (3,))
    # Image Augmentation
    x = layers.RandomContrast(factor=0.2, seed=config.SEED)(input)
    x = layers.RandomFlip(mode='horizontal_and_vertical', seed=config.SEED)(x)
    x = layers.RandomRotation(factor=1.0, seed=config.SEED)(x)
    x = layers.RandomZoom(height_factor=(-0.2, 0.2),
                          width_factor=(-0.2, 0.2), seed=config.SEED)(x)
    x = layers.RandomCrop(height=config.CROP,
                          width=config.CROP, seed=config.SEED)(x)
    # EfficientNetB3 expects images to be in the range [0, 255]
    x = enb3(x)
    # Classifier head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(rate=0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    mdl = Model(inputs=input, outputs=output)
    mdl.compile(optimizer=Nadam(learning_rate=config.LEARNING_RATE),
                loss=losses.BinaryCrossentropy(),
                metrics=[metrics.BinaryAccuracy(),
                         metrics.AUC(),
                         metrics.Precision(),
                         metrics.Recall()])
    return mdl


def run_training() -> None:
    """Train the model."""

    # read training data
    train = load_dataset()
    validation = load_dataset(subset='validation')

    learn_control = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=5,
        verbose=0,
        restore_best_weights=True)

    # train model
    mdl = model()
    mdl.fit(train, epochs=config.EPOCHS, validation_data=validation,
            callbacks=[learn_control])

    with open(BASE_PATH / 'VERSION') as version_file:
        save_file_name = 'model' + version_file.read().strip() + '.pkl'
    joblib.dump(mdl, MODEL_PATH / save_file_name)


if __name__ == "__main__":
    run_training()
