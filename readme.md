# Программа для анализа fMRI данных

Программа предназначена для предобработки фМРТ-данных, обучения модели и получения предсказаний. Все параметры указываются в YAML конфигурации. Путь к конфигу можно передать при запуске `main.py`. Если аргумент не указан, используется файл `test_config.yaml` из корня проекта.

---

## Формат конфигурационного файла

Ниже приведён минимальный пример структуры конфига:

```yaml
atlas_path: "./atlas/atlas_resample.nii"
train_model: true           # обучать модель или использовать готовую
model_path: "./models/model.pkl"
save_to_numpy: false        # сохранять ли промежуточные npy-файлы
func: "max"                 # метод агрегации признаков
features_type: "ranks_five" # тип извлекаемых признаков
results_path: "./results.txt"

training_dataset:
  - data_path: ./data/train/subj1/sdenoised_data.nii.gz
    events_path: ./data/train/subj1/events.txt
    tr: 1.0
    numpy_path: ./numpy_data/train/subj1.npy
  - ...

test_dataset:
  - data_path: ./data/test/subjA/sdenoised_data.nii.gz
    events_path: ./data/test/subjA/events.txt
    tr: 1.0
    numpy_path: ./numpy_data/test/subjA.npy
  - ...
```

- `training_dataset` и `test_dataset` содержат списки испытуемых.
- Если `train_model` равно `true`, модель будет обучена на данных из `training_dataset`. В противном случае она будет загружена из `model_path`.
- Итоговые предсказания сохраняются в файл, указанный в `results_path`.

---

## Запуск программы

1. Установите зависимости:

```bash
pip3 install -r requirements.txt
```

2. Запустите скрипт, указав путь к своему конфигу (необязательно):

```bash
python3 main.py path/to/config.yaml
```

Если путь не указан, будет использован `test_config.yaml`.


