'''
    Скрипт, который считывает данные согласно конфигу и затем подает модели и записывает результаты в файл.
    TODO: сделать ревью скрипта
'''

import pickle
from fmri_processing import *


atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'
config_path = '/home/aaanpilov/diploma/project/config_test_data.yaml'


if __name__ == '__main__':
    # Получаем даннные из конфига    
    subjects = process_config(config_path)

    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Создаем загрузчик данных
    data_loader = DataLoader(atlas_path)

    need_average = True

    with open('results.txt', 'w') as f:
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
            processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_max_min, need_average=need_average)

            if need_average:
                # Проверяем данные на модели
                predicted_truth = model.predict(processed_truth.reshape(1, 132))
                predicted_lie = model.predict(processed_lie.reshape(1, 132))
                print(predicted_truth)
                print(predicted_lie)
                
            else:
                # Проверяем данные на модели
                predicted_truth = model.predict(processed_truth)
                predicted_lie = model.predict(processed_lie)

                str = f'{subject["data_path"]}\n'
                # Выводим результаты
                str += get_predict_results_str(sub.get_events(), predicted_truth, predicted_lie)
                str += '\n'
                f.write(str)
                
