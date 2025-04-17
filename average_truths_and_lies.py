import pickle
from fmri_processing import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'
config_path = '/home/aaanpilov/diploma/project/config_HC.yaml'




def visualize(data):
# Создаем тепловую карту
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Значения'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта массива')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.show()



if __name__ == '__main__':
    # Получаем даннные из конфига    
    subjects = process_config(config_path)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = True

    truth_matrix = None
    lie_matrix = None


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

        if truth_matrix is None:
            truth_matrix = processed_truth.reshape(1, -1)
        else:
            truth_matrix = np.concatenate((truth_matrix, processed_truth.reshape(1, -1)), axis=0)
        
        if lie_matrix is None: 
            lie_matrix = processed_lie.reshape(1, -1)
        else:
            lie_matrix = np.concatenate((lie_matrix, processed_lie.reshape(1, -1)), axis=0)

    # print(truth_matrix == lie_matrix)
    matrix_truth_lie = np.concatenate((truth_matrix, lie_matrix), axis=0)        
    visualize(matrix_truth_lie)
    # np.save('truth_lie_matrix_test', matrix_truth_lie)

