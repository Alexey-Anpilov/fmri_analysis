from fmri_processing import *
import numpy as np
from fmri_processing.utils import draw_heat_map 
import os 
from enum import Enum, auto

atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'

class DataOption(Enum):
    RAW_HC = auto()
    RAW_TEST = auto()
    HC = auto()
    TEST = auto()


def build_average_matrix(config_path, matrix_path, process_func):
    # Получаем даннные из конфига    
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    need_average = True
    truth_matrix = None
    lie_matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            data = data_loader.load_from_npy(subject['numpy_path'])
        else:
            data = data_loader.load_from_nii_and_save(subject['numpy_path'])

        # Получаем и обрабатываем данные

        events = data_loader.load_events(subject['events_path'])
        if data is None or events is None:
            continue

        # Формируем объект хранящий данные
        sub = SubjectData()
        sub.set_data(data)
        sub.set_events(events)
        sub.set_tr(subject['tr'])
        
        # Обрезаем и преобразуем данные
        processed_truth, processed_lie = sub.cut_for_truth_and_lie(window_size=10 , process_func=process_func, need_average=need_average)

        if truth_matrix is None:
            truth_matrix = processed_truth.reshape(1, -1)
        else:
            truth_matrix = np.concatenate((truth_matrix, processed_truth.reshape(1, -1)), axis=0)
        
        if lie_matrix is None: 
            lie_matrix = processed_lie.reshape(1, -1)
        else:
            lie_matrix = np.concatenate((lie_matrix, processed_lie.reshape(1, -1)), axis=0)

    matrix_truth_lie = np.concatenate((truth_matrix, lie_matrix), axis=0)        
    draw_heat_map(matrix_truth_lie)
    np.save(matrix_path, matrix_truth_lie)


if __name__ == '__main__':
    save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_matrix'

    config_path_raw_hc = '/home/aaanpilov/diploma/project/configs/raw_hc_data.yaml'
    raw_hc = 'raw_HC'

    config_path_hc = '/home/aaanpilov/diploma/project/configs/hc_data.yaml'
    hc = 'HC'

    config_path_raw_test = '/home/aaanpilov/diploma/project/configs/raw_test_data.yaml'
    raw_test = 'raw_test'

    config_path_test = '/home/aaanpilov/diploma/project/configs/test_data.yaml'
    test = 'test'

    for data_option in DataOption:
        if data_option == DataOption.HC:
            config_path = config_path_hc
            data_option_path = hc
        elif data_option == DataOption.RAW_HC:
            config_path = config_path_raw_hc
            data_option_path = raw_hc
        elif data_option == DataOption.RAW_TEST:
            config_path = config_path_raw_test
            data_option_path = raw_test
        else:
            config_path = config_path_test
            data_option_path = test

        for name, func in funcs.items():
            matrix_path = os.path.join(os.path.join(save_dir, data_option_path), name)
            build_average_matrix(config_path, matrix_path, func)