import os

def extract_brain_mask_files(folder):
    files = os.listdir(folder)
    
    mask = [os.path.join(folder, file) for file in files if file.endswith('MASK.nii.gz') or file.endswith('MASK_EDIT.nii.gz')]
    img = [os.path.join(folder, file) for file in files if file.endswith('FL.nii.gz') or file.endswith('T1.nii.gz') or file.endswith('T2.nii.gz')]
    
    return img, mask
    