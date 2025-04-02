import os
import numpy as np
import pickle

from fmri_processing import *
from fmri_processing.subjects_info import *




def load_from_nii_and_save(data_loader):
    '''
        Сохраняет npy матрицы для фМРТ данных в отдельной директории
    '''
    for sub_id in sub_info:
        fmri_path = os.path.join(data_dir, sub_id, data_file)
        output_path = os.path.join(out_dir, f"{sub_id}.npy")
        data_loader.load_from_nii_and_save(fmri_path, output_path)


def load_from_npy(data_loader):
    '''
        Загружает фМРТ данные из npy и из файлов с разбиением в локальные переменные 
    '''
    subjects_data = dict()

    for sub_id in sub_info:
        fmri_path = os.path.join(out_dir, f"{sub_id}.npy")
        events_path = os.path.join(data_dir, sub_id, events_file)

        sub = SubjectData()
        sub.set_tr(sub_info[sub_id]['tr'])
        data_loader.load_data(sub, fmri_path, events_path, True, True)

        subjects_data[sub_id] = sub
    
    return subjects_data 


# def process_and_build_matrix(self, process_func, output_dir):
#     subjects_results = []
#     for sub_id in self.sub_info:
#         truth_data, lie_data = self.cut_answers_for_subject(sub_id)
#         if truth_data is None:
#             continue
#         avg_truth, avg_lie = self.process_subject_(truth_data, lie_data, process_func)
#         subjects_results.append((avg_truth, avg_lie))
#     # Построение матрицы
#     matrix = self.
# (subjects_results)
#     np.save(output_dir, matrix)   # Тут надо добавить аутпут директорию
#     print("Матрица данных:")
#     print(matrix.shape)  # (2 * n_subjects, 132)




def build_matrix(subjects_results):
    """
    Собирает матрицу из результатов всех испытуемых.

    Параметры:
    - subjects_results: Список кортежей (avg_truth, avg_lie) для каждого испытуемого

    Возвращает:
    - matrix: Матрица формы (n_subjects * 2, 132)
            Первая половина строк — правда, вторая — ложь.
    """
    n_subjects = len(subjects_results)
    matrix = np.zeros((2 * n_subjects, 132))

    # Заполнение матрицы
    for i, (avg_truth, avg_lie) in enumerate(subjects_results):
        matrix[i] = avg_truth          # Строки 0..n_subjects-1: правда
        matrix[n_subjects + i] = avg_lie  # Строки n_subjects..2n_subjects-1: ложь

    return matrix


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

config_path = '/home/aaanpilov/diploma/project/config_HC.yaml'


if __name__ == '__main__':
    subjects = process_config(config_path)
     # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = True

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



