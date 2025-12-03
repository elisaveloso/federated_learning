import os

import settings


if not os.path.exists(settings.DATASETS_BASE_DIR):
    os.makedirs(settings.DATASETS_BASE_DIR)

if not os.path.exists(settings.DATASET_DIR):
    os.makedirs(settings.DATASET_DIR)


for class_type in settings.CLASSES_DICT[settings.DATASET_TYPE]:
    class_dir = os.path.join(settings.DATASET_DIR, class_type)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
