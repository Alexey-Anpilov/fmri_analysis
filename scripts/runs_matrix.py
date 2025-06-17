import numpy as np
from enum import Enum, auto
from fmri_processing import *
from fmri_processing.subjects_info import *
from fmri_processing.utils import draw_heat_map
import numpy as np
import os
from fmri_processing.utils import *
from pathlib import Path

atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'

class DataOption(Enum):
    RAW_HC = auto()
    RAW_TEST = auto()
    RAW_SCHZ = auto()
    RAW_CARD_HC = auto()
    RAW_CARD_TEST = auto()
    HC = auto()
    TEST = auto()
    SCHZ = auto()
    CARD_HC = auto()
    CARD_TEST = auto()


configs = {
    DataOption.RAW_HC : '/home/aaanpilov/diploma/project/configs/raw_hc_data.yaml',
    DataOption.RAW_TEST : '/home/aaanpilov/diploma/project/configs/raw_test_data.yaml',
    # DataOption.RAW_SCHZ : '/home/aaanpilov/diploma/project/configs/raw_schz_data.yaml',
    # DataOption.RAW_CARD_HC : '/home/aaanpilov/diploma/project/configs/raw_card_hc_data.yaml',
    # DataOption.RAW_CARD_TEST : '/home/aaanpilov/diploma/project/configs/raw_card_test_data.yaml',
    # DataOption.HC : '/home/aaanpilov/diploma/project/configs/hc_data.yaml',
    # DataOption.TEST : '/home/aaanpilov/diploma/project/configs/test_data.yaml',
    # DataOption.SCHZ : '/home/aaanpilov/diploma/project/configs/schz_data.yaml',
    # DataOption.CARD_HC : '/home/aaanpilov/diploma/project/configs/card_hc_data.yaml',
    # DataOption.CARD_TEST : '/home/aaanpilov/diploma/project/configs/card_test_data.yaml',
}


# Для чемпиона 
def get_stats(ranks_list, stimulus_index=3):
    # Инициализация
    counts_3_5 = np.zeros(132, dtype=int)  # Счётчик для 132 регионов

    # Перебор всех прогонов (runs)
    for ranks in ranks_list:
        is_five = (ranks[stimulus_index, :] >= 4)  # Где стимул 3 == 5 в текущем прогоне
        counts_3_5 += is_five.astype(int)

    return counts_3_5.reshape(1, -1)

# Пропорциональные баллы
def normalize_proportional(data):
    # Копируем данные, чтобы не изменять исходный массив
    data = np.array(data, dtype=float)
    ranks = np.zeros_like(data)
    
    for region in range(data.shape[1]):
        # Получаем значения для текущего региона
        region_data = data[:, region]
        
        # Вычисляем сумму значений (для нормализации)
        sum_values = np.sum(region_data)
        
        if sum_values > 0:
            # Распределяем баллы пропорционально значениям
            # Сумма баллов должна быть 15 (1+2+3+4+5)
            proportional_scores = 15 * region_data / sum_values
        else:
            # Если все значения нулевые, ставим равные баллы
            proportional_scores = np.ones(5) * 3  # Среднее значение
            
        # Округляем до 2 знаков после запятой (для читаемости)
        ranks[:, region] = np.round(proportional_scores, 2)
    
    return ranks

# Баллы
def normalize(data):
# Ваш массив (5 стимулов × 132 региона)

    # Для каждого региона (столбца) получаем ранги стимулов
    ranks = np.zeros_like(data, dtype=int)

    for region in range(data.shape[1]):
        # Получаем индексы, которые сортируют значения в столбце (от меньшего к большему)
        sorted_indices = np.argsort(data[:, region])
        # Преобразуем индексы в ранги (1 для минимального, 6 для максимального)
        ranks[sorted_indices, region] = np.arange(1, 6)  # 1..6
    return ranks

# Баллы 1-2
def normalize_reduced(data):
    # Создаем массив нулей того же размера, что и входные данные
    ranks = np.zeros_like(data, dtype=int)
    
    for region in range(data.shape[1]):
        # Получаем индексы, которые сортируют значения в столбце (от меньшего к большему)
        sorted_indices = np.argsort(data[:, region])
        # Присваиваем:
        # - 0 для всех элементов по умолчанию (уже сделано при создании ranks)
        # - 1 для второго по величине
        ranks[sorted_indices[-2], region] = 1
        # - 2 для максимального
        ranks[sorted_indices[-1], region] = 2
    
    return ranks


# Расстановка баллов
def process_runs_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
            data = data_loader.load_from_npy(numpy_path)

        # Получаем и обрабатываем данные
        events = data_loader.load_events(subject['events_path'])
        if data is None or events is None:
            continue

        # Формируем объект хранящий данные
        sub = SubjectData()
        sub.set_data(data)
        sub.set_events(events)
        sub.set_tr(subject['tr'])

        runs = sub.cut_for_runs(window_size=10)
        ranks_list = list()
        for run in runs:
            processed_data = sub.apply_func(run, processing_func)
            ranks = normalize_func(processed_data)
            ranks_list.append(ranks)
        summed_ranks = np.sum(ranks_list, axis=0)

        if matrix is None:
            matrix = summed_ranks
        else:
            matrix = np.concatenate((matrix, summed_ranks), axis=0)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)


# Определение чемпиона
def process_runs_comparison_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
            data = data_loader.load_from_npy(numpy_path)

        events = data_loader.load_events(subject['events_path'])
        if data is None or events is None:
            continue

        # Формируем объект хранящий данные
        sub = SubjectData()
        sub.set_data(data)
        sub.set_events(events)
        sub.set_tr(subject['tr'])

        runs = sub.cut_for_runs(window_size=10)
        ranks_list = list()
        for run in runs:
            processed_data = sub.apply_func(run, processing_func)
            ranks = normalize_func(processed_data)
            ranks_list.append(ranks)
        
        stats_matrix = None
        for i in range(len(ranks_list)):
            stimulus_stats = get_stats(ranks_list, i)
            if stats_matrix is None:
                stats_matrix = stimulus_stats
            else:
                stats_matrix = np.concatenate((stats_matrix, stimulus_stats), axis=0)
        if matrix is None:
            matrix = stats_matrix
        else:
            matrix = np.concatenate((matrix, stats_matrix), axis=0)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)


# Обрабатываем стимулы отдельно
def process_stimulus_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
            data = data_loader.load_from_npy(numpy_path)

        events = data_loader.load_events(subject['events_path'])
        if data is None or events is None:
            continue

        # Формируем объект хранящий данные
        sub = SubjectData()
        sub.set_data(data)
        sub.set_events(events)
        sub.set_tr(subject['tr'])

        runs = sub.cut_for_runs(window_size=10)
        stimulus_data = np.mean(runs, axis=0)
        
        processed_stimulus_data = sub.apply_func(stimulus_data, processing_func)

        if matrix is None:
            matrix = processed_stimulus_data
        else:
            matrix = np.concatenate((matrix, processed_stimulus_data), axis=0)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)    


# Обрабатываем стимулы отдельно
def process_stimulus_with_different_trials_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
            data = data_loader.load_from_npy(numpy_path)

        events = data_loader.load_events_for_different_trials(subject['events_path'])

        if data is None or events is None:
            continue
        
        for events_pack in events:
            # Формируем объект хранящий данные
            sub = SubjectData()
            sub.set_data(data)
            sub.set_events(events_pack)
            sub.set_tr(subject['tr'])

            runs = sub.cut_for_runs(window_size=10)
            stimulus_data = np.mean(runs, axis=0)
            
            processed_stimulus_data = sub.apply_func(stimulus_data, processing_func)

            if matrix is None:
                matrix = processed_stimulus_data
            else:
                matrix = np.concatenate((matrix, processed_stimulus_data), axis=0)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)    
    print(matrix.shape)


# Усредняем правдивые и ложные ответы отдельно
def process_runs_into_average_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader()

    matrix = None

    # Инициализация списков с помощью спискового включения
    matrix_list_truth = [None] * 5
    matrix_list_lie = [None] * 5

    
    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
            data = data_loader.load_from_npy(numpy_path)
            
        events = data_loader.load_events(subject['events_path'])
        if data is None or events is None:
            continue

        # Формируем объект хранящий данные
        sub = SubjectData()
        sub.set_data(data)
        sub.set_events(events)
        sub.set_tr(subject['tr'])

        runs = sub.cut_for_runs(window_size=10)
        stimulus_list = list()
        for run in runs:
            processed_data = sub.apply_func(run, processing_func)
            stimulus_list.append(processed_data)

        for k in range(len(stimulus_list)):  # Перебираем каждую строку (0, 1, 2, 3, 4)
            # 1. Усредняем k-ю строку всех массивов
            k_row_avg = np.mean([arr[k] for arr in stimulus_list], axis=0, keepdims=True)  # (1, 132)
            
            # 2. Усредняем все остальные строки (все, кроме k-й)
            other_rows = np.concatenate([np.delete(arr, k, axis=0) for arr in stimulus_list])  # (4*N, 132)
            other_avg = np.mean(other_rows, axis=0, keepdims=True)  # (1, 132)
            
            # Обновляем матрицы truth и lie с помощью более компактного кода
            matrix_list_truth[k] = other_avg if matrix_list_truth[k] is None else np.concatenate((matrix_list_truth[k], other_avg))
            matrix_list_lie[k] = k_row_avg if matrix_list_lie[k] is None else np.concatenate((matrix_list_lie[k], k_row_avg
                                                                                              ))

    for k in range(len(matrix_list_truth)):
        matrix = np.concatenate((matrix_list_truth[k], matrix_list_lie[k]))

        if k == 3:
            draw_heat_map(matrix)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        np.save(matrix_path + str(k), matrix)



def build_matrixes_for_all(save_dir, processing_func ,normalize_func=normalize):
    for option, config in configs.items():
        for name, func in funcs.items():
            data_option_path = Path(config).stem
            matrix_path = os.path.join(os.path.join(save_dir, data_option_path), name)
            processing_func(config, matrix_path, func, normalize_func)


def build_ranks_matrixes(save_dir,normalize_func=normalize):
    build_matrixes_for_all(save_dir, process_runs_and_save_matrix, normalize_func)


def build_stimulus_matrixes(save_dir):
    build_matrixes_for_all(save_dir, process_stimulus_and_save_matrix)


if __name__ == '__main__':
    #--------------------------------------------------------------------------------------------------------------------
    # ВСЕ ДЛЯ РАНГОВ
    save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix'  
    build_ranks_matrixes(save_dir)

    save_dir_proportional_ranks = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/proportional'
    build_ranks_matrixes(save_dir_proportional_ranks, normalize_proportional)

    save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/reduced_ranks'  
    build_ranks_matrixes(save_dir, normalize_reduced)

    #--------------------------------------------------------------------------------------------------------------------
    # УСРЕДНЕННЫЙ СТИМУЛ
    save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_stimulus'  
    build_stimulus_matrixes(save_dir)

    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_stimulus'  
    # config = configs[DataOption.CARD_HC]
    # for name, func in funcs.items():
    #     data_option_path = Path(config).stem
    #     matrix_path = os.path.join(os.path.join(save_dir, data_option_path), name)
    #     process_stimulus_with_different_trials_and_save_matrix(config, matrix_path, func, normalize)


    #--------------------------------------------------------------------------------------------------------------------
    # УСРЕДНЕННОЕ ВСЕ
    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_matrix/propose'  
    # for name, func in funcs.items():
    #     matrix_path = os.path.join(os.path.join(save_dir, 'test'), name)
    #     process_runs_into_average_matrix('/home/aaanpilov/diploma/project/configs/test_data.yaml', matrix_path, func)

    #--------------------------------------------------------------------------------------------------------------------
    # ЧЕМПИОН
    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/prizes'  
    # build_ranks_matrixes_for_all(save_dir, normalize)
    #--------------------------------------------------------------------------------------------------------------------














    # def build_for_one(save_dir, normalize_func=normalize):
    # config_card_hc = '/home/aaanpilov/diploma/project/configs/card_hc.yaml'
    # data_option_path = 'card_hc'

    # # config_schz = '/home/aaanpilov/diploma/project/configs/schz.yaml'
    # # data_option_path = 'schz'

    # config_path = config_card_hc
    
    # for name, func in funcs.items():
    #     matrix_path = os.path.join(os.path.join(save_dir, data_option_path), name)
    #     process_runs_and_save_matrix(config_path, matrix_path, func, normalize_func)
    #     # process_runs_comparison_and_save_matrix(config_path, matrix_path, func, normalize_func)
    #     # process_runs_into_params_and_save_matrix(config_path, matrix_path, func)
