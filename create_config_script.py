import os
import yaml

# Путь к директории с данными
base_dir = "./test_data"
output_config = "config_test_data.yaml"  # Имя выходного файла

subjects = []

# Проходим по всем поддиректориям в test_data
for dir_name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, dir_name)
    
    # Пропускаем файлы (если есть), работаем только с папками
    if not os.path.isdir(dir_path):
        continue
    
    # Ищем .txt файл в папке (берём первый найденный)
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
    print(txt_files)
    if not txt_files:
        print(f"Внимание: в папке {dir_name} нет .txt файла!")
        continue
    
    # Формируем запись для конфига
    subject_data = {
        'data_path': os.path.join(dir_path, 'sdenoised_data.nii.gz'),
        'events_path': os.path.join(dir_path, txt_files[0]),
        'tr': 1.1,
        'numpy_path': os.path.join('./numpy_data/test_data', f'{dir_name}.npy') 
    }
    subjects.append(subject_data)

# Создаём структуру конфига
config = {'subjects': subjects}

# Сохраняем в YAML
with open(output_config, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Конфиг успешно создан: {output_config}")