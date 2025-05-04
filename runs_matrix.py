import numpy as np
from enum import Enum, auto
from fmri_processing import *
from fmri_processing.subjects_info import *
from fmri_processing.utils import draw_heat_map
import numpy as np
import os
import matplotlib.pyplot as plt
from fmri_processing.utils import *

class DataOption(Enum):
    RAW_HC = auto()
    RAW_TEST = auto()
    HC = auto()
    TEST = auto()


def get_stats(ranks_list, stimulus_index=3):
    # Инициализация
    counts_3_5 = np.zeros(132, dtype=int)  # Счётчик для 132 регионов

    # Перебор всех прогонов (runs)
    for ranks in ranks_list:
        is_five = (ranks[stimulus_index, :] >= 4)  # Где стимул 3 == 5 в текущем прогоне
        counts_3_5 += is_five.astype(int)

    return counts_3_5.reshape(1, -1)
    # # Анализ результатов
    # max_count = np.max(counts_3_5)
    # top_regions = np.where(counts_3_5 == max_count)[0]

    # print(f"Третий стимул чаще всего был максимальным в регионах: {top_regions}")
    # print(f"Максимальное количество раз: {max_count}")

    # # Визуализация (опционально)
    # plt.figure(figsize=(12, 6))
    # plt.bar(range(132), counts_3_5)
    # plt.xlabel("Регион")
    # plt.ylabel("Количество прогонов с рангом 5")
    # plt.title("Частота максимального ранга для стимула 3")
    # plt.show()


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


def process_runs_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
        else:
            numpy_path = None

        # Получаем и обрабатываем данные
        data = data_loader.load_data(subject['data_path'], numpy_path)
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
        summed_ranks = np.sum(ranks_list, axis=0)  # форма (6, 132)

        if matrix is None:
            matrix = summed_ranks
        else:
            matrix = np.concatenate((matrix, summed_ranks), axis=0)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)


def process_runs_comparison_and_save_matrix(config_path, matrix_path, processing_func=calc_auc, normalize_func=normalize):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
        else:
            numpy_path = None

        # Получаем и обрабатываем данные
        data = data_loader.load_data(subject['data_path'], numpy_path)
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



def process_runs_into_params_and_save_matrix(config_path, matrix_path, processing_func=calc_auc):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    matrix = None

    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
        else:
            numpy_path = None

        # Получаем и обрабатываем данные
        data = data_loader.load_data(subject['data_path'], numpy_path)
        print(data.shape)
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

        draw_heat_map(processed_stimulus_data)

        if matrix is None:
            matrix = processed_stimulus_data
        else:
            matrix = np.concatenate((matrix, processed_stimulus_data), axis=0)
    print(matrix.shape)
    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
    np.save(matrix_path, matrix)    


def process_runs_into_average_matrix(config_path, matrix_path, processing_func=calc_auc):
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    matrix = None


    # Инициализация списков с помощью спискового включения
    matrix_list_truth = [None] * 5
    matrix_list_lie = [None] * 5

    
    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        # Проверяем есть ли путь к сохраненной numpy матрице
        if 'numpy_path' in subject:
            numpy_path = subject['numpy_path']
        else:
            numpy_path = None

        # Получаем и обрабатываем данные
        data = data_loader.load_data(subject['data_path'], numpy_path)
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
        print(matrix.shape)
        if k == 3:
            draw_heat_map(matrix)
        os.makedirs(os.path.dirname(matrix_path), exist_ok=True)
        np.save(matrix_path + str(k), matrix)


def build_ranks_matrixes_for_all(save_dir, normalize_func=normalize):
    config_path_raw_hc = '/home/aaanpilov/diploma/project/configs/raw_HC_data.yaml'
    raw_hc = 'raw_HC'

    config_path_hc = '/home/aaanpilov/diploma/project/configs/HC_data.yaml'
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
            # process_runs_and_save_matrix(config_path, matrix_path, func, normalize_func)
            # process_runs_comparison_and_save_matrix(config_path, matrix_path, func, normalize_func)
            process_runs_into_params_and_save_matrix(config_path, matrix_path, func)

def build_for_one(save_dir, normalize_func=normalize):
    config_card_hc = '/home/aaanpilov/diploma/project/configs/card_hc.yaml'
    data_option_path = 'card_hc'

    # config_schz = '/home/aaanpilov/diploma/project/configs/schz.yaml'
    # data_option_path = 'schz'

    config_path = config_card_hc
    
    for name, func in funcs.items():
        matrix_path = os.path.join(os.path.join(save_dir, data_option_path), name)
        process_runs_and_save_matrix(config_path, matrix_path, func, normalize_func)
        # process_runs_comparison_and_save_matrix(config_path, matrix_path, func, normalize_func)
        # process_runs_into_params_and_save_matrix(config_path, matrix_path, func)


if __name__ == '__main__':
    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix'  
    # build_ranks_matrixes_for_all(save_dir, normalize)

    # save_dir_proportional_ranks = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/proportional'
    # build_ranks_matrixes_for_all(save_dir_proportional_ranks, normalize_proportional)

    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_stimulus'  
    # build_ranks_matrixes_for_all(save_dir, normalize)

    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/reduced_ranks'  
    # build_ranks_matrixes_for_all(save_dir, normalize_reduced)

    # draw_heat_map(np.load('/home/aaanpilov/diploma/project/numpy_matrixes/average_matrix/test/auc.npy'))

    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/average_matrix/propose'  
    # for name, func in funcs.items():
    #     matrix_path = os.path.join(os.path.join(save_dir, 'test'), name)
    #     process_runs_into_average_matrix('/home/aaanpilov/diploma/project/configs/test_data.yaml', matrix_path, func)


    # save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix/prizes'  
    # build_ranks_matrixes_for_all(save_dir, normalize)

    save_dir = '/home/aaanpilov/diploma/project/numpy_matrixes/ranks_matrix' 
    build_for_one(save_dir)