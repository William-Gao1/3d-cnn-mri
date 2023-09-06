import os

import nibabel as nib
import numpy as np

import albumentations as A
import cv2

import keras
import tensorflow as tf

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dir, extract_files_for_subject, num_in_channels = 2, in_dim=(128, 128, 128), out_dim=(128, 128, 128), mode='train_with_aug'):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_in_channels = num_in_channels
        self.extract_files_for_subject = extract_files_for_subject
        self.folders = [os.path.join(dir, x) for x in os.listdir(dir)]
        self.on_epoch_end()
        self.data_path = dir
        
        if mode == 'train_with_aug':
            self.augmentations = A.Compose([
                A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Resize(128, 128, always_apply=True)
            ], additional_targets={f'channel_{i}': 'image' for i in range(num_in_channels - 1)})
        else:
            self.augmentations = A.Compose([
                A.Resize(128, 128)
            ], additional_targets={f'channel_{i}': 'image' for i in range(num_in_channels - 1)})
            
    def __len__(self):
        return len(self.folders)
    
    def __getitem__(self, index):
        folder_index = self.indexes[index]
        x, y = self.__data_generation(self.folders[folder_index])
        
        return x, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.folders))
        np.random.shuffle(self.indexes)
        
    def __data_generation(self, folder):
        # each batch is 1 image for now
        X = np.zeros((*self.in_dim, self.num_in_channels))
        
        inputs, labels = self.extract_files_for_subject(folder)
        
        assert len(inputs) == self.num_in_channels, inputs
        assert len(labels) == 1 # only 1 out channel for now
        
        input_imgs = []
        label = None
        
        transformed_imgs = None
        
        for input_img in inputs:
            img_data = nib.load(input_img).get_fdata()
            if img_data.shape[2] == 256:
                img_data = img_data[..., ::2]
            img_data = img_data / np.max(img_data)
            input_imgs.append(img_data)
            
        if type(labels[0]) == str:
            label = nib.load(labels[0]).get_fdata()
            if label.shape[2] == 256:
                label = label[..., ::2]
            transformed_imgs = self.augmentations(image=input_imgs[0], mask = label, **{f'channel_{i}': img for i, img in enumerate(input_imgs[1:])})
            Y = np.zeros((1, *self.in_dim, 2)) # only 1 out channel for now (2 channels after one hot)
            Y[0] = tf.one_hot(transformed_imgs["mask"], 2) # binary mask
        else:
            label = labels[0]
            transformed_imgs = self.augmentations(image=input_imgs[0], **{f'channel_{i}': img for i, img in enumerate(input_imgs[1:])})
            #Y[0] = tf.one_hot(label, 2) # binary input
            Y = np.zeros((1, 1)) # only 1 out channel for now
            Y[0] = label
        
        X[..., 0] = transformed_imgs["image"]
        
        for i in range(0, len(input_imgs) - 1):
            X[..., i + 1] = transformed_imgs[f'channel_{i}']
        
        # need another dimension to make keras happy :) (i think it's for batches)
        X_final = np.zeros((1, *self.in_dim, self.num_in_channels))
        X_final[0] = X
        return X_final, Y
            
        