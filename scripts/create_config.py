'''
    Скрипт, который позволяет создавать конфиги по директории с данными.
    Директория должна содержать поддиректории, в которых находятся данные 
    и файлы с временным разбиением. TODO: добавить псевдографику с деревом директории.
'''




import os
import yaml
from fmri_processing import subjects_info


# Путь к директории с данными
base_dir = "/home/aaanpilov/diploma/project/data/test_data"
output_config = "/home/aaanpilov/diploma/project/configs/raw_test_data.yaml"  # Имя выходного файла
data_file_name = 'denoised_data.nii.gz'
numpy_data_dir = '/home/aaanpilov/diploma/project/numpy_data/raw_test_data'
sub_info = subjects_info.sub_info_hc


# TODO: Тут можно переделать, чтобы принимал еще sub-info

subjects = []

# Проходим по всем поддиректориям в test_data
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    
    # Пропускаем файлы (если есть), работаем только с папками
    if not os.path.isdir(dir_path):
        continue
    
    # Ищем .txt файл в папке (берём первый найденный)
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    print('\n' in txt_files[0])
    if not txt_files:
        print(f"Внимание: в папке {dir_name} нет .txt файла!")
        continue
    
    # Формируем запись для конфига
    subject_data = {
        'data_path': os.path.join(dir_path, data_file_name),
        'events_path': os.path.join(dir_path, txt_files[0]),
        'tr': 1.11,
        'numpy_path': os.path.join(numpy_data_dir, f'{dir_name}.npy') 
    }
    subjects.append(subject_data)

# Создаём структуру конфига
config = {'subjects': subjects}

# Сохраняем в YAML
with open(output_config, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Конфиг успешно создан: {output_config}")