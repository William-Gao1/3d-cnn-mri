from models.create_model import create_model
from data_gen.generator import DataGenerator

import keras
import keras.backend as K

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', required=True)
parser.add_argument('--test-dir', required=True)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', required=True)
parser.add_argument('--name', required=True)
parser.add_argument('--without-augmentations', default=True, type=bool)

args = parser.parse_args()

# TRAIN_DATASET_PATH = '/hpf/projects/ndlamini/scratch/wgao/skull_strip_fl/train'
# TEST_DATASET_PATH = '/hpf/projects/ndlamini/scratch/wgao/skull_strip_fl/test'

callbacks = [
      keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.000001, verbose=1),
]

model, extraction_function, n_in_channels = create_model(args.model, 128)
print(model.summary())
print(f'Training mode: {"train_without_aug" if args.without_augmentations else "train_with_aug"}')
training_generator = DataGenerator(args.train_dir, extraction_function, num_in_channels=n_in_channels, mode="train_without_aug" if args.without_augmentations else "train_with_aug")
test_generator = DataGenerator(args.test_dir, extraction_function, num_in_channels=n_in_channels, mode="test")

K.clear_session()

model.fit(training_generator, 
          epochs=args.epochs,
          steps_per_epoch=len(training_generator),
          callbacks=callbacks,
          validation_data=test_generator
)

model.save(args.name)