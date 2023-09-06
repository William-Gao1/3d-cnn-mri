#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=william.gao@sickkids.ca
#SBATCH -c 1
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 12:00:00

module load python/3.8.0

export PYTHONPATH=/hpf/projects/ndlamini/scratch/wgao/python3.8.0
export LD_LIBRARY_PATH=$PYTHONPATH/nvidia/cudnn/lib:$PYTHONPATH/tensorrt_libs:/hpf/tools/centos7/cuda/11.2/lib64:$LD_LIBRARY_PATH

#python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/nienke_process/test' --model 'lesion_symptom_img' --name 'lsm.keras'
#python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/aabdalla/ml/acute_stroke_training/train' --test-dir '/hpf/projects/ndlamini/scratch/aabdalla/ml/acute_stroke_training/test' --model 'stroke_segmentation' --name 'stroke_seg_dwi.keras'
#python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t2/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/skull_strip_t2/test' --model 'brain_mask' --name 't2_strip.keras'
python3 train.py --train-dir '/hpf/projects/ndlamini/scratch/wgao/b1000_seg_data/train' --test-dir '/hpf/projects/ndlamini/scratch/wgao/b1000_seg_data/train' --model 'stroke_segmentation' --name 'stroke_seg_b1000_final.keras'

