from .brain_mask_model import create_brain_mask_model
from .lesion_symptom_model import create_lesion_symptom_img_model
from .stroke_segmentation_model import create_stroke_segmentation_model
from data_gen.extraction import extract_brain_mask_files, extract_stroke_segmentation_files, extract_lsm_files_1_channel, extract_lsm_files_2_channel

def get_model(model_name, img_size):
    # this function returns the keras network, the extraction function, and the number of in channels
    if model_name == 'brain_mask':
        return create_brain_mask_model(img_size), extract_brain_mask_files, 1
    elif model_name == 'lesion_symptom_2_channel':
        return create_lesion_symptom_img_model(img_size, 2), extract_lsm_files_2_channel, 2
    elif model_name == 'lesion_symptom_1_channel':
        return create_lesion_symptom_img_model(img_size), extract_lsm_files_1_channel, 1
    elif model_name == 'stroke_segmentation':
        return create_stroke_segmentation_model(img_size), extract_stroke_segmentation_files, 2