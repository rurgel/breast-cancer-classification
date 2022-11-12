from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data import Dataset
from pathlib import Path
from typing import Tuple

from detect_breast_cancer.config.core import config


def load_dataset(*,
                 subset: str = 'training',
                 input_path: Path = config.INPUT_PATH,
                 image_size: Tuple[int, int] = config.IMAGE_SIZE,
                 batch_size: int = config.BATCH_SIZE,
                 validation_split=config.VALIDATION_SPLIT,
                 seed=config.SEED
                 ) -> Dataset:
    ds = image_dataset_from_directory(
        input_path,
        validation_split=validation_split,
        subset=subset,
        seed=seed,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
    )
    return ds
