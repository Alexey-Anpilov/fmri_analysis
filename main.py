from fmri_processing import *
import argparse
import os
import sys
import numpy as np
import pickle
from sklearn.metrics import recall_score, classification_report

def parse_arguments(default_config_path="test_config.yaml"):
    """
    Парсит аргументы командной строки и возвращает путь к конфигурационному файлу
    
    Параметры:
    default_config_path (str): Путь по умолчанию к конфигурационному файлу
    
    Возвращает:
    str: Абсолютный путь к конфигурационному файлу
    
    Выбрасывает исключения:
    SystemExit: При невалидных аргументах или отсутствии файла
    """
    parser = argparse.ArgumentParser(
        description="Обработчик конфигурации для проекта анализа фМРТ данных"
    )
    
    # Добавляем аргумент для пути к конфигурационному файлу
    parser.add_argument(
        'config_path',
        nargs='?',  # Делаем аргумент необязательным
        default=default_config_path,
        help=f"Путь к конфигурационному YAML-файлу (по умолчанию: {default_config_path})"
    )
    
    # Парсим аргументы
    args = parser.parse_args()
    
    # Преобразуем в абсолютный путь
    config_path = os.path.abspath(args.config_path)
    
    # Проверяем существование файла
    if not os.path.isfile(config_path):
        print(f"ОШИБКА: Конфигурационный файл не найден: {config_path}", file=sys.stderr)
        print(f"Текущая рабочая директория: {os.getcwd()}", file=sys.stderr)
        sys.exit(1)
    
    return config_path


if __name__ == "__main__":
    try:
        config_path = parse_arguments()
        print(f"Используется конфигурационный файл: {config_path}")
        
        # Здесь можно вызвать вашу функцию загрузки конфига
        config = load_and_validate_config(config_path)
        
        processing_func = funcs[config['func']]

        if config['train_model']:
            matrix_builder = FeatureMatrixBuilder()
            train_subs = config['training_dataset']
            train_matrix = matrix_builder.build_matrix(config, train_subs, processing_func)
            test_subs = config['test_dataset']
            test_matrix = matrix_builder.build_matrix(config, test_subs, processing_func)
            y_train = np.zeros(train_matrix.shape[0], dtype=int)
            y_train[3::5] = 1 
            model = StimulusClassifier()
            model.fit(train_matrix, y_train, config['model_path'])

            y_test = np.zeros(test_matrix.shape[0], dtype=int)
            y_test[3::5] = 1 
            
            print(classification_report(y_test, model.predict(test_matrix)))
        else:
            matrix_builder = FeatureMatrixBuilder()
            test_subs = config['test_dataset']
            test_matrix = matrix_builder.build_matrix(config, test_subs, processing_func)

            # Загрузка модели
            with open(config['model_path'], 'rb') as f:
                model = pickle.load(f)

            X = test_matrix
            y_test = np.zeros(X.shape[0], dtype=int)
            y_test[3::5] = 1 

            probabilities = model.predict_proba(X)[:, 1] 
            final_predictions = np.zeros(len(X))

            for i in range(0, len(X), 5):
                # Индексы текущего испытуемого
                subject_indices = range(i, i + 5)
                # Индекс стимула с максимальной P(class=1)
                most_confident_idx = np.argmax(probabilities[subject_indices]) + i
                final_predictions[most_confident_idx] = 1
            
            predictions_matrix = final_predictions.reshape(-1, 5)
            print("Итоговые предсказания:\n", predictions_matrix)

            results_path = config['results_path']
            with open(results_path, 'w') as f:
                for idx, subject in enumerate(test_subs):
                    data_file = subject.get('numpy_path') or subject.get('data_path')
                    preds = predictions_matrix[idx]
                    f.write(f"{data_file}\n")
                    f.write(' '.join(map(str, preds.astype(int))) + "\n")

    except SystemExit:
        sys.exit(1)

