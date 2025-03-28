from yaml import safe_load
import numpy as np

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



# TODO просмотреть вот эту функцию 
def print_predict_results(events, predicted_truth, predicted_lie):
    truth_array = np.column_stack((events[events['trial_type'] == 0]['onset'].values, predicted_truth, events[events['trial_type'] == 0]['trial_type'].values))
    lie_array = np.column_stack((events[events['trial_type'] == 1]['onset'].values, predicted_lie, events[events['trial_type'] == 1]['trial_type'].values))
    
    combined_array = np.vstack((truth_array, lie_array))

    sorted_array = combined_array[combined_array[:, 0].argsort()]

    for elem in sorted_array:
        status = "содержит сокрытие информации" if elem[1] == 1 else "не содержит сокрытия информации"
        print(f"{elem[0]} : {status} {elem[2]}")