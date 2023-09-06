# 3D CNN Templates

This repositiory holds the code for some 3D CNNs on MRI images. Currently there are 3D CNNs for:

- Automatic Skull Stripping
- Automatic Stroke Segmentation
- Lesion Symptom Prediction

## Usage/Examples

To train a network, modify the `submit.sh` file:

- Skull stripping network
  - T1:
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t2/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t2/test' --model 'brain_mask' --name 't2_strip.keras'
    ```
  - T2:
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t1/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t1/test' --model 'brain_mask' --name 't1_strip.keras'
    ```
  - FL:
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_fl/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_fl/test' --model 'brain_mask' --name 'fl_strip.keras'
    ```
- Stroke segmentation network
  - DWI + ADC:
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/aabdalla/ml/acute_stroke_training/train' --test-dir '/hpf/projects/ndlamini/scratch/aabdalla/ml/acute_stroke_training/test' --model 'stroke_segmentation' --name 'stroke_seg_dwi.keras'
    ```
  - b1000 + ADC:
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/b1000_seg_data/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/b1000_seg_data/train' --model 'stroke_segmentation' --name 'stroke_seg_b1000.keras'
    ```
- LSM:
  - One channel with augmentations
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/test' --model 'lesion_symptom_1_channel' --name 'lsm_1_channel.keras'
    ```
  - One channel without augmentations
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/test' --model 'lesion_symptom_1_channel' --name 'lsm_1_channel_wo_aug.keras' --without-augmentations
    ```
  - Two channels with augmentations
    ```bash
    python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/test' --model 'lesion_symptom_2_channel' --name 'lsm_2_channel.keras'
    ```

Then, to start training, run `sbatch submit.sh`

## Contributing

The 3D network input has shape (128, 128, 128) and has output of (128, 128, 128). But for consistency with the 2D networks, the images in the train and test dataset need to be (256, 256, 256). The code will automatically resize the images down to (128, 128, 128)

Each of the train and test dataset should have the following structure:

```
train/
├── subject1/
│   ├── subject1_T1.nii.gz
│   ├── subject1_MASK.nii.gz
│   └── ...
├── subject2/
│   ├── subject2_T1.nii.gz
│   ├── subject2_MASK.nii.gz
│   └── ...
└── ...
```

In `data_gen/extraction`, there should be a function that picks out the inputs and labels given a subject folder. If the labels are images, it should return a list of paths to those images. If the labels are numbers, (e.g. lesion symptom scores), it should return the numbers. See `data_gen/extraction/stroke_segmentation_extraction.py` and `data_gen/extraction/stroke_segmentation_extraction.py` for a well documented examples.

The blocks that are common to all networks are in `models/layers/layers.py`. These blocks are used in `models/architectures` to create the model
