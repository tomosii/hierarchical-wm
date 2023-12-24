import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os


class BouncingBalls(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Dataset of bouncing balls sequences."),
            features=tfds.features.FeaturesDict(
                {
                    "video": tfds.features.Video(shape=(35, 32, 32, 3)),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        paths = {"npy_file": "path/to/your/npy_file.npy"}
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"file_path": paths["npy_file"], "split": "train"},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={"file_path": paths["npy_file"], "split": "test"},
            ),
        ]

    def _generate_examples(self, file_path, split):
        # Load the data from the .npy file
        data = np.load(file_path, allow_pickle=True)

        # Decide the split index
        split_index = int(len(data) * 0.8)  # Assuming 80% for training, 20% for testing

        # Split the data into train and test sets
        if split == "train":
            split_data = data[:split_index]
        elif split == "test":
            split_data = data[split_index:]

        # Yield each video sequence
        for i, sequence in enumerate(split_data):
            yield i, {"video": sequence}


# Load the dataset
# train_dataset = tfds.load('bouncing_balls', split='train', builder_kwargs={"config": "your_config_if_any"})
# test_dataset = tfds.load('bouncing_balls', split='test', builder_kwargs={"config": "your_config_if_any"})
