import os
from typing import List, Tuple

# remember to add this function to __init__.py!
def extract_stroke_segmentation_files(folder: str) -> Tuple[List[str], List[str]]:
    """Given a subject folder containing files for stroke segmentation, return the images and labels

    Args:
        folder (str): Full path to subject folder

    Returns:
        Tuple[List[str], List[str]]: A tuple whose first element is a list of full paths of the DWI, ADC and the second
                                        element being the full path to the mask
    """ 
    files = os.listdir(folder)
    
    masks = [os.path.join(folder, file) for file in files if file.endswith('SEG.nii.gz')]
    dwis = [os.path.join(folder, file) for file in files if file.endswith('DWI.nii.gz') or file.endswith('B1000.nii.gz')]
    adcs = [os.path.join(folder, file) for file in files if file.endswith('ADC.nii.gz')]
    assert len(dwis) == 1
    assert len(adcs) == 1
    assert len(masks) == 1
    
    return [dwis[0], adcs[0]], [masks[0]]
    