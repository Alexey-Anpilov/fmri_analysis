from yaml import safe_load
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

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
    data=data[:5]
    plt.figure(figsize=(10, 6))  # Задаем размер графика
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': 'Сумма баллов'})  # 'viridis' — цветовая карта
    plt.title('Тепловая карта признаков')
    plt.xlabel('Регионы')
    plt.ylabel('Стимулы')
    plt.show()


def load_and_validate_config(config_path: str) -> dict:
    """
    Загружает и валидирует YAML-конфигурационный файл с новой структурой
    
    Параметры:
    config_path (str): Путь к YAML-конфигурационному файлу
    
    Возвращает:
    dict: Валидированный конфигурационный словарь
    
    Выбрасывает исключения:
    - FileNotFoundError: Если файл не существует
    - ValueError: При нарушении валидационных правил
    - KeyError: При отсутствии обязательных ключей
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл {config_path} не найден")
    
    with open(config_path, 'r') as f:
        config = safe_load(f)
    
    # Проверка обязательных глобальных полей
    required_global = ['atlas_path', 'train_model', 'model_path', 
                      'save_to_numpy', 'features_type', 'func']
    for key in required_global:
        if key not in config:
            raise KeyError(f"Отсутствует обязательное поле: {key}")
    
    # Проверка существования atlas файла
    atlas_path = Path(config['atlas_path'])
    if not atlas_path.exists():
        raise FileNotFoundError(f"Atlas файл не найден: {atlas_path}")
    
    # Валидация значений func
    valid_funcs = ['max', 'auc', 'min', 'max-min']
    if config['func'] not in valid_funcs:
        raise ValueError(
            f"Недопустимое значение func: {config['func']}. "
            f"Допустимые значения: {valid_funcs}"
        )
    
    # Валидация значений features_type
    valid_features = ['average_stimulus', 'ranks_five', 'ranks_two']
    if config['features_type'] not in valid_features:
        raise ValueError(
            f"Недопустимое значение features_type: {config['features_type']}. "
            f"Допустимые значения: {valid_features}"
        )
    
    # Проверка пути к модели
    if not config['model_path']:
        raise ValueError("Путь к модели (model_path) не может быть пустым")
    
    # Проверка наличия хотя бы одного блока данных
    if 'training_dataset' not in config and 'testing_dataset' not in config:
        raise KeyError("Конфиг должен содержать хотя бы один блок: training_dataset или testing_dataset")
    
    # Проверка необходимости обучающей выборки
    if config['train_model']:
        if 'training_dataset' not in config:
            raise KeyError("При train_model=true требуется блок training_dataset")
        
        if not config['training_dataset'] or not isinstance(config['training_dataset'], list):
            raise ValueError("Блок training_dataset должен быть списком с данными")
        
        if len(config['training_dataset']) == 0:
            raise ValueError("Блок training_dataset должен содержать хотя бы одного субъекта")
    
    # Валидация структуры блоков данных
    for block in ['training_dataset', 'testing_dataset']:
        if block in config and config[block] is not None:
            if not isinstance(config[block], list):
                raise ValueError(f"Блок {block} должен быть списком")
            
            # Проверка обязательных полей для каждого субъекта
            for i, subject in enumerate(config[block]):
                required_subject_keys = ['data_path', 'events_path', 'tr', 'numpy_path']
                for key in required_subject_keys:
                    if key not in subject:
                        raise KeyError(
                            f"Субъект #{i+1} в блоке {block} "
                            f"не содержит обязательный ключ: {key}"
                        )
                
                # Проверка существования файлов
                for path_key in ['data_path', 'events_path']:
                    path = Path(subject[path_key])
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Файл не найден: {path} "
                            f"(субъект #{i+1} в блоке {block})"
                        )
    
    return config