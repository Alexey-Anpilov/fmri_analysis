from fmri_processing import *
import os

import numpy as np
import pickle


sub_info_hc = {
'sub-01' : {'slice_num' : 546, 'tr' : 1.0},
'sub-02' : {'slice_num' : 553, 'tr' : 1.0},
'sub-03' : {'slice_num' : 457, 'tr' : 1.0},
'sub-04' : {'slice_num' : 492, 'tr' : 1.0},
'sub-05' : {'slice_num' : 929, 'tr' : 1.0},
'sub-06' : {'slice_num' : 478, 'tr' : 1.0},
'sub-07' : {'slice_num' : 546, 'tr' : 1.0},
'sub-08' : {'slice_num' : 468, 'tr' : 1.11},
'sub-09' : {'slice_num' : 433, 'tr' : 1.11},
'sub-10' : {'slice_num' : 462, 'tr' : 1.11},
'sub-11' : {'slice_num' : 441, 'tr' : 1.11},
'sub-12' : {'slice_num' : 451, 'tr' : 1.11},
'sub-13' : {'slice_num' : 416, 'tr' : 1.11},
'sub-14' : {'slice_num' : 419, 'tr' : 1.11},
'sub-15' : {'slice_num' : 420, 'tr' : 1.11},
'sub-16' : {'slice_num' : 453, 'tr' : 1.11},
'sub-17' : {'slice_num' : 404, 'tr' : 1.11},
'sub-18' : {'slice_num' : 471, 'tr' : 1.11}
}


sub_info_schz = {
'sub-01' : { 'slice_num' : 423, 'tr' : 1.11},
'sub-02' : { 'slice_num' : 524, 'tr' : 1.0},
'sub-03' : { 'slice_num' : 618, 'tr' : 1.0},
'sub-04' : { 'slice_num' : 502, 'tr' : 1.0},
'sub-05' : { 'slice_num' : 485, 'tr' : 1.0},
'sub-06' : { 'slice_num' : 531, 'tr' : 1.0},
'sub-07' : { 'slice_num' : 551, 'tr' : 1.0},
'sub-08' : { 'slice_num' : 491, 'tr' : 1.0},
'sub-09' : { 'slice_num' : 744, 'tr' : 1.1},
'sub-10' : { 'slice_num' : 399, 'tr' : 1.1},
'sub-11' : { 'slice_num' : 513, 'tr' : 1.1},
'sub-12' : { 'slice_num' : 472, 'tr' : 1.1},
'sub-13' : { 'slice_num' : 409, 'tr' : 1.1},
'sub-14' : { 'slice_num' : 426, 'tr' : 1.1},
'sub-15' : { 'slice_num' : 472, 'tr' : 1.0},
'sub-16' : { 'slice_num' : 431, 'tr' : 1.11},
'sub-17' : { 'slice_num' : 441, 'tr' : 1.11},
'sub-18' : { 'slice_num' : 495, 'tr' : 1.11},
'sub-19' : { 'slice_num' : 449, 'tr' : 1.11},
}


black_list_hc = ['sub-01', 'sub-07']
black_list_schz = ['sub-01', 'sub-17', 'sub-16', 'sub-15']

atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'
data_file = 'denoised_data.nii.gz'
events_file = "time_file.csv"

data_dir_hc = './data/HC'
out_dir_hc = './numpy_data/HC/'

data_dir_schz = './data/SCHZ/'
out_dir_schz = './numpy_data/SCHZ/'



is_hc = True

if is_hc:
     sub_info = sub_info_hc
     black_list = black_list_hc
     data_dir = data_dir_hc
     out_dir = out_dir_hc
else:
    sub_info = sub_info_schz
    black_list = black_list_schz
    data_dir = data_dir_schz
    out_dir = out_dir_schz


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
        data_loader.load_data_for_sub(sub, fmri_path, events_path, True, True)

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

if __name__ == '__main__':
    data_loader = DataLoader(atlas_path)
    subjects_data = load_from_npy(data_loader)

    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    need_average = False

    processed_subjects_true = np.empty((0, 132))
    processed_subjects_lie = np.empty((0, 132))
    
    for sub_id, sub in subjects_data.items():
        if sub_id in black_list:
            continue
        # Обрезаем и преобразуем данные
        processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_max_min, need_average=need_average)
        
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



