import os

import settings
import ml_dataset_prep

classes_list = settings.CLASSES_DICT[settings.DATASET_TYPE]

ml_dataset_prep.remove_corrupted_images(settings.DATASET_DIR)

ml_dataset_prep.balance_dataset(settings.DATASET_DIR, classes_list)

ml_dataset_prep.compress_dataset(settings.DATASET_DIR)



    
