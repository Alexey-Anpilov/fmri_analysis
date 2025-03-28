import numpy as np
import pickle
from yaml import safe_load
from fmri_processing import *


atlas_path = '/home/aaanpilov/diploma/project/atlas/atlas_resample.nii'
config_path = 'config.yaml'


if __name__ == '__main__':
    # Получаем даннные из конфига    
    data_path, events_path, tr = process_config(config_path)

    # Получаем фМРТ данные и разбиваем на регионы
    sub = SubjectData()
    sub.set_tr(tr)
    data_loader = DataLoader(atlas_path)
    data_loader.load_data_for_sub(sub, data_path, events_path)
    # TODO: тут можно вместо функции принимающей sub, сделать то, что возвращает
    # также сделать просто две фукнции, одна берет данные из npy или nii
    # вторая берет данные из csv или оригинального файла

    # Обрезаем и преобразуем данные
    processed_truth, processed_lie = sub.cut_and_apply_function(window_size=10 , process_func=calc_max_min)
    
    # Загружаем модель
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Проверяем данные на модели
    predicted_truth = model.predict(processed_truth)
    predicted_lie = model.predict(processed_lie)

    # Выводим результаты
    print_predict_results(sub.get_events(), predicted_truth, predicted_lie)
