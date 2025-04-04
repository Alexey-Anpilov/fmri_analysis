import numpy as np
import pickle

from fmri_processing import *
from fmri_processing.subjects_info import *

config_path = './config_HC.yaml'

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


if __name__ == '__main__':
    subjects = process_config(config_path)
     # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = False

    is_first = True
    processed_subjects_true = np.empty((0, 132))
    processed_subjects_lie = np.empty((0, 132))
    # Интерируемся по объектам в конфиге
    for subject in subjects:        
        if is_first:
            is_first=False
            continue
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
        
        for run in sub.cut_for_runs(window_size=50):
            draw_signal(run)
            visualize(calc_auc(run))

        # Обрезаем и преобразуем данные
        processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_auc, need_average=need_average)

        if need_average:
            processed_subjects_true = np.concatenate((processed_subjects_true, processed_truth.reshape(1, 132)))
            processed_subjects_lie = np.concatenate((processed_subjects_lie, processed_lie.reshape(1, 132)))
        else: 
            processed_subjects_true = np.concatenate((processed_subjects_true, processed_truth))
            processed_subjects_lie = np.concatenate((processed_subjects_lie, processed_lie))
    


    if need_average:
        np.save('./results/HC/area_matrix', np.concatenate((processed_subjects_true, processed_subjects_lie)))
    else:
        np.save('./results/HC/max_matrix_true', processed_subjects_true)
        np.save('./results/HC/max_matrix_lie', processed_subjects_lie)
