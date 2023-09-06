
import sys
import os

sys.path.insert(0, '/hpf/projects/ndlamini/scratch/wgao/python3.8.0/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras.models
import nibabel
import albumentations as A
import numpy as np
from nibabel import processing

model_name = sys.argv[1]
channel_1_file = sys.argv[2]
channel_2_file = None
if len(sys.argv) == 4:
    channel_2_file = sys.argv[3]

model = keras.models.load_model(model_name, compile=False)

p = None

def predict_2_channel(channel_1_path, channel_2_path):
    global p
    
    channel_1_img = nibabel.load(channel_1_path)
    channel_2_img = nibabel.load(channel_2_path)
    
    # conform to 256 x 256 x 256
    channel_1_img_resampled = processing.conform(channel_1_img)
    channel_2_img_resampled = processing.conform(channel_2_img)
    
    # take every second image to go from 256 slices -> 128 slices
    channel_1_img_voxels = channel_1_img_resampled.get_fdata()[:, :, ::2]
    channel_2_img_voxels = channel_2_img_resampled.get_fdata()[:, :, ::2]
    
    # normalize by dividing by max value and resize to 128 x 128
    channel_1_img_voxels = A.resize(channel_1_img_voxels/np.max(channel_1_img_voxels), 128, 128)
    channel_2_img_voxels = A.resize(channel_2_img_voxels/np.max(channel_2_img_voxels), 128, 128)
    
    X = np.zeros((1, 128, 128, 128, 2))
    X[0, :, :, :, 0] = channel_1_img_voxels
    X[0, :, :, :, 1] = channel_2_img_voxels
    
    p = model.predict(X, verbose=1)

def predict_1_channel(channel_1_path):
    global p
    
    channel_1_img = nibabel.load(channel_1_path)
    
    # conform to 256 x 256 x 256
    channel_1_img_resampled = processing.conform(channel_1_img)
    
    # take every second image to go from 256 slices -> 128 slices
    channel_1_img_voxels = channel_1_img_resampled.get_fdata()[:, :, ::2]
    
    # normalize by dividing by max value and resize to 128 x 128
    channel_1_img_voxels = A.resize(channel_1_img_voxels/np.max(channel_1_img_voxels), 128, 128)
    
    X = np.zeros((1, 128, 128, 128, 1))
    X[0, :, :, :, 0] = channel_1_img_voxels
    
    p = model.predict(X, verbose=1)

if channel_2_file is not None:
    predict_2_channel(channel_1_file, channel_2_file)
else:
    predict_1_channel(channel_1_file)

np.save('pred.npy', p)

print('Prediction saved to pred.npy')