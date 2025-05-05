'''
    Скрипт, который позволяет создавать конфиги по директории с данными.
    Директория должна содержать поддиректории, в которых находятся данные 
    и файлы с временным разбиением. TODO: добавить псевдографику с деревом директории.
'''
import os
import yaml
from fmri_processing import subjects_info


def create_config(base_dir, data_file_name, output_config_path=None, numpy_data_dir=None, tr_info=None):
    subjects = []

    # Устанавливаем значения по умолчанию
    if tr_info is None:
        tr_info = {}
    
    dir_names = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))])
    
    # Проходим по всем поддиректориям в test_data
    for dir_name in dir_names:
        dir_path = os.path.join(base_dir, dir_name)
        
        # Пропускаем файлы (если есть), работаем только с папками
        if not os.path.isdir(dir_path):
            continue
        
        # Ищем .txt файл в папке (берём первый найденный)
        txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

        if not txt_files:
            print(f"Внимание: в папке {dir_name} нет .txt файла!")
            continue
        
        if len(txt_files) > 1:
            print(f"В диретории {dir_name} более 1 txt файла. Проверьте, что в конфиг попал корректный файл")

        txt_file = txt_files[0]

        # Формируем запись для конфига
        subject_data = {
            'data_path': os.path.join(dir_path, data_file_name) if data_file_name else '',
            'events_path': os.path.join(dir_path, txt_file),
            'tr': tr_info.get(dir_name, 1.0),  # Используем значение из tr_info или 1.0 по умолчанию
            'numpy_path': os.path.join(numpy_data_dir, f'{dir_name}.npy') if numpy_data_dir else ''
        }
        subjects.append(subject_data)

    # Создаём структуру конфига
    config = {'subjects': subjects}

    # Сохраняем в YAML
    if output_config_path:
        with open(output_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Конфиг успешно создан: {output_config_path}")
    else:
        print("Внимание: output_config_path не указан, конфиг не сохранен на диск")


if __name__ == '__main__':
    base_dir = '/home/aaanpilov/diploma/project/data'
    base_config_dir = '/home/aaanpilov/diploma/project/configs'
    base_numpy_dir = '/home/aaanpilov/diploma/project/numpy_data'
    data_file_name = 'sdenoised_data.nii.gz'

    for name in ('hc_data', 'test_data', 'schz_data', 'card_hc_data', 'card_test_data'):
        # Создаем конфиги для z-score данных
        create_config(
            base_dir=os.path.join(base_dir, name),
            data_file_name=data_file_name,
            output_config_path=os.path.join(base_config_dir, name+'.yaml'),
            numpy_data_dir=os.path.join(base_numpy_dir, name),
            tr_info = subjects_info.tr_info[name]
        )

        raw_name = 'raw_' + name
        # Создаем конфиги для сырых данных
        create_config(
            base_dir=os.path.join(base_dir, name),
            data_file_name=data_file_name,
            output_config_path=os.path.join(base_config_dir, raw_name + '.yaml'),
            numpy_data_dir=os.path.join(base_numpy_dir, raw_name),
            tr_info = subjects_info.tr_info[name]
        )