import os

from typing import List, Tuple

def extract_lsm_files_1_channel(folder: str) -> Tuple[List[str], List[int]]:
    files = os.listdir(folder)
    
    masks = [os.path.join(folder, file) for file in files if file.endswith('SEG.nii.gz')]
    imgs = [os.path.join(folder, file) for file in files if file.endswith('IMG.nii.gz')]
    scores = [os.path.join(folder, file) for file in files if file.endswith('.txt')]
    
    assert len(masks) == 1 or len(imgs) == 1
    assert len(scores) == 1
        
    with open(scores[0], 'r') as f:
        score = int(f.readline())
    
    if len(imgs) > 0:
        return [imgs[0]], [score]
    else:
        return [masks[0]], [score]

def extract_lsm_files_2_channel(folder: str) -> Tuple[List[str], List[int]]:
    files = os.listdir(folder)
    
    masks = [os.path.join(folder, file) for file in files if file.endswith('SEG.nii.gz')]
    t1s = [os.path.join(folder, file) for file in files if file.endswith('IMG.nii.gz')]
    scores = [os.path.join(folder, file) for file in files if file.endswith('.txt')]
    
    assert len(masks) == 1
    assert len(scores) == 1
    assert len(t1s) == 1
    
    with open(scores[0], 'r') as f:
        score = int(f.readline())
    
    return [t1s[0], masks[0]], [score]
    