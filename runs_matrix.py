import numpy as np
import pickle

from fmri_processing import *
from fmri_processing.subjects_info import *

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import seaborn as sns
import matplotlib.pyplot as plt
def visualize(data):
# Создаем тепловую карту
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Значения'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта массива (480, 132)')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def draw_signal(data):
    # Выбираем регион (например, регион с индексом 0)
    selected_regions = [0]

  # Создаём сетку графиков:
    fig, axes = plt.subplots(
        nrows=data.shape[0],  # Количество графиков = x
        figsize=(10, 3 * data.shape[0]),  # Размер фигуры
        sharex=True  # Общая ось X для удобства
    )

    # Если x=1, axes станет scalar, поэтому приводим к списку:
    if data.shape[0] == 1:
        axes = [axes]

    # Рисуем каждый график:
    for i, ax in enumerate(axes):
        for region in selected_regions:
            ax.plot(data[i, :, region], label=f'Регион {region}')
        ax.set_ylabel('Активность')
        ax.set_title(f'График {i + 1}')
        ax.grid(True)
        ax.legend()

    plt.xlabel('Время')
    plt.tight_layout()  # Чтобы подписи не накладывались
    plt.show()

def visualize_region(data):
    roi_signal = data[:, 0]  # Индексы начинаются с 0!

# Визуализация
    plt.figure(figsize=(12, 6))
    plt.plot(roi_signal, label=f'Регион {0}', color='red')
    plt.title(f'Временной ряд для региона {0}')
    plt.xlabel('Временные точки')
    plt.ylabel('Сигнал (нормализованный)')
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize(data):
# Ваш массив (6 стимулов × 132 региона)

    # Для каждого региона (столбца) получаем ранги стимулов
    ranks = np.zeros_like(data, dtype=int)

    for region in range(data.shape[1]):
        # Получаем индексы, которые сортируют значения в столбце (от меньшего к большему)
        sorted_indices = np.argsort(data[:, region])
        # Преобразуем индексы в ранги (1 для минимального, 6 для максимального)
        ranks[sorted_indices, region] = np.arange(1, 6)  # 1..6
    return ranks


def process_runs_and_save_matrix(config_path, matrix_path):
    subjects = process_config(config_path)
     # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = False

    matrix = None

    processed_subjects_true = np.empty((0, 132))
    processed_subjects_lie = np.empty((0, 132))
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
            processed_data = sub.apply_func(run, calc_minimum)
            ranks_list.append(normalize(processed_data))
        summed_ranks = np.sum(ranks_list, axis=0)  # форма (6, 132)

        if matrix is None:
            matrix = summed_ranks
        else:
            matrix = np.concatenate((matrix, summed_ranks), axis=0)
    np.save(matrix_path, matrix)

def process_runs_for_and_save_matrix(config_path, matrix_path):
    subjects = process_config(config_path)
    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = False

    matrix = None

    processed_subjects_true = np.empty((0, 132))
    processed_subjects_lie = np.empty((0, 132))
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
            processed_data = sub.apply_func(run, calc_auc)
            print(processed_data.shape)
            ranks_list.append(processed_data)
            mean_area = np.mean(ranks_list, axis=0)  # форма (6, 132)
            zscore_scaler = StandardScaler()
            mean_area = zscore_scaler.fit_transform(mean_area)
        if matrix is None:
            matrix = mean_area
        else:
            matrix = np.concatenate((matrix, mean_area), axis=0)
    np.save(matrix_path, matrix)





if __name__ == '__main__':  
    test_config_path = './config_test_data.yaml'
    config_path = './config_raw_HC_data.yaml'

    paths = [
        {
            'config':  './config_raw_HC_data.yaml',
            'matrix': './ranks_matrix/HC_raw_matrix_min' 
         },
         {
            'config': './config_test_data.yaml',
            'matrix': './ranks_matrix/test_matrix_min' 
         }
    ]


    paths_area = [
        {
            'config':  './config_raw_HC_data.yaml',
            'matrix': './area_matrix/HC_raw_matrix_auc' 
         },
         {
            'config': './config_test_data.yaml',
            'matrix': './area_matrix/test_matrix_auc' 
         }
    ]

    for path in paths:
        process_runs_and_save_matrix(path['config'], path['matrix'])
        # process_runs_for_and_save_matrix(path['config'], path['matrix'])



        # for key, item in res.items():
        #     processed_data = sub.apply_func(np.array(item), calc_auc)


        #     if draw_data is None:
        #         draw_data = processed_data[:,:1]
        #     else:
        #         draw_data = np.concatenate((draw_data, processed_data[:, :1]), axis=1)
        # plt.figure(figsize=(16, 4))

# # Построение тепловой карты
#         heatmap = sns.heatmap(
#         draw_data,                   # Данные
#         annot=False,            # Не показывать значения в ячейках (если True — может быть слишком много)
#         cmap="viridis",         # Цветовая схема ("coolwarm", "plasma", "magma", "YlOrRd")
#         yticklabels=[f"Строка {i+1}" for i in range(5)],  # Подписи строк
#         cbar=True,              # Показать цветовую шкалу
#         linewidths=0.5,         # Тонкие границы между ячейками
#         linecolor="grey"        # Цвет границ
#         )

#         plt.title("Тепловая карта всех строк (5 × 132)")
#         plt.xlabel("Номер столбца")
#         plt.ylabel("Строки")
#         plt.show()
    
        
        # Обрезаем и преобразуем данные
    #     processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_auc, need_average=need_average)

    #     if need_average:
    #         processed_subjects_true = np.concatenate((processed_subjects_true, processed_truth.reshape(1, 132)))
    #         processed_subjects_lie = np.concatenate((processed_subjects_lie, processed_lie.reshape(1, 132)))
    #     else: 
    #         processed_subjects_true = np.concatenate((processed_subjects_true, processed_truth))
    #         processed_subjects_lie = np.concatenate((processed_subjects_lie, processed_lie))
    


    # if need_average:
    #     np.save('./results/HC/area_matrix', np.concatenate((processed_subjects_true, processed_subjects_lie)))
    # else:
    #     np.save('./results/HC/max_matrix_true', processed_subjects_true)
    #     np.save('./results/HC/max_matrix_lie', processed_subjects_lie)
