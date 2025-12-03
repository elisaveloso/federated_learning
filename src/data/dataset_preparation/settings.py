import os
from datetime import date

DATASET_TYPE =  'daninhas'

DATASETS_BASE_DIR = '/home/elisaveloso/federated_learning/src/data/datasets'

#Data de criação do dataset
DATE = date.today().strftime("%y-%m-%d")

#Pasta de criação do dataset
DATASET_DIR = os.path.join(DATASETS_BASE_DIR, f'{DATASET_TYPE}__{DATE}')

#Dicionário com a as classes para cada tipo de dataset
CLASSES_DICT = {
    'daninhas': ['daninha', 'nao_daninha']
}