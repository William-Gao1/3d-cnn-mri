from models.architectures import get_model
from models.metrics.metrics import dice_coef, precision, sensitivity, specificity

import keras

def create_model(model_name, img_size, learning_rate=0.0001):
    model, extraction_function, n_in_channels = get_model(model_name, img_size)
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    metrics = []
    if 'lesion_symptom' in model_name:
        metrics = [
            keras.metrics.BinaryAccuracy(threshold=0.5)
        ]
    else:
        metrics = [
            'accuracy',
            dice_coef,
            precision,
            sensitivity,
            specificity,
            keras.metrics.MeanIoU(num_classes=2)
        ]
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)
    
    return model, extraction_function, n_in_channels