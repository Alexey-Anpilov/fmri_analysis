from lib import *

import numpy as np
import pickle
from yaml import safe_load


atlas_path = './atlas/atlas_resample.nii'


def process_config(config_path):
    with open(config_path) as f:
        data = safe_load(f)
        data_file = data['data_path']
        events_file = data['events_path']
        tr = data['tr']
        return data_file, events_file, tr


# def process_events(events_file):
#     events_data = {'onset': [], 'duration': [], 'trial_type': []}  # Создаем словарь для хранения данных

#     with open(events_file, 'r') as input_file:
#         for line in input_file:
#             l = line.split()  # Разделяем строку на части
#             if len(l) == 3:  # Пропускаем строки с неправильным форматом
#                 continue
#             if l[3][2] == '4':  # Проверяем условие для trial_type
#                 trial_type = '1'
#             else:
#                 trial_type = '0'

#             # Добавляем данные в словарь
#             events_data['onset'].append(float(l[1]))  # Преобразуем onset в float
#             events_data['duration'].append(1.0)      # Фиксированная длительность
#             events_data['trial_type'].append(trial_type)  # Тип события

#     return pd.DataFrame(events_data)


def process_events(file_name):
    with open('./tmp.csv', 'w') as output_file,\
        open(file_name, 'r') as input_file:
            output_file.write('onset,duration,trial_type\n')
            for line in input_file:
                l = line.split()
                if len(l) == 3:
                    continue
                if l[3][2] == '4':
                    output_file.write(l[1] + ',1.0,'  +  '1'   + '\n')
                else:
                    output_file.write(l[1] + ',1.0,' + '0' + '\n')
    events_data = pd.read_csv('./tmp.csv')

    return events_data


def build_matrix_from_data(data_file, events, tr):
    masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=True, verbose=1)
    fmri_data = masker.fit_transform(data_file)
    data_processor = DataProcessor(fmri_data, events)
    return data_processor.cut_answers(fmri_data, events, tr)


def maximum(data):
    return np.max(data, axis=1)


def minimum(data):
    return np.min(data, axis=1)


def max_min(data):
    return np.max(data, axis=1) - np.min(data, axis=1)


def calculate_auc(data):
    return np.trapz(data, axis=1)  # Интегрируем по времени для каждого ответа и региона


if __name__ == '__main__':
    config_path = 'config.yaml'
    process_config(config_path)

    # Получаем даннные из конфига    
    data_file, events_file, tr = process_config(config_path)

    # Обрабатываем файл с разметкой
    events = process_events(events_file)
    
    # Преобразуем данные
    truth_data, lie_data = build_matrix_from_data(data_file, events, tr)
    
    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    area_truth, area_lie = calculate_auc(truth_data), calculate_auc(lie_data)


    truth_array = np.column_stack((events[events['trial_type'] == 0]['onset'].values, model.predict(area_truth)))
    lie_array = np.column_stack((events[events['trial_type'] == 1]['onset'].values, model.predict(area_lie)))
    
    combined_array = np.vstack((truth_array, lie_array))

    sorted_array = combined_array[combined_array[:, 0].argsort()]

    for elem in sorted_array:
        status = "содержит сокрытие информации" if elem[1] == 1 else "не содержит сокрытия информации"
        print(f"{elem[0]} : {status}")
