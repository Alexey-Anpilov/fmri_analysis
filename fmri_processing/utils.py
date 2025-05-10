from yaml import safe_load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def process_config(config_path):
    with open(config_path) as f:
        data = safe_load(f)
        subjects = data['subjects']
        return subjects


# TODO просмотреть вот эту функцию 
def get_predict_results_str(events, predicted_truth, predicted_lie):
    truth_array = np.column_stack((events[events['trial_type'] == 0]['onset'].values, predicted_truth, events[events['trial_type'] == 0]['trial_type'].values))
    lie_array = np.column_stack((events[events['trial_type'] == 1]['onset'].values, predicted_lie, events[events['trial_type'] == 1]['trial_type'].values))
    
    combined_array = np.vstack((truth_array, lie_array))

    sorted_array = combined_array[combined_array[:, 0].argsort()]

    res = ''
    for elem in sorted_array:
        status = "содержит сокрытие информации" if elem[1] == 1 else "не содержит сокрытия информации"
        res += f"{elem[0]} : {status} {elem[2]} \n"
    
    return res 


def draw_heat_map(data):
    '''
        Создает тепловую карту для данных размера (x, 132)
    '''
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Значения'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта массива')
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.show()