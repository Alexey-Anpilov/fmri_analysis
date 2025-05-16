import logging
from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


"""
Класс DataLoader предназначен для:
1. Загрузки и предобработки fMRI данных из NIfTI-файлов с использованием атласа
2. Сохранения/загрузки предобработанных данных в формате .npy
3. Загрузки и обработки данных о событиях из текстовых файлов
"""
class DataLoader:
    def __init__(self):
        self.logger = self._setup_logger()


    def _setup_logger(self):
        """Инициализирует и настраивает логгер для класса"""
        logger = logging.getLogger("DataLoader")
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
            )
            logger.addHandler(handler)
        return logger


    def load_from_nii_and_save(
        self,
        data_path,
        npy_path,
        tr,
        atlas_path,
        standardize=False
    ):
        """
        Обработка NIfTI файла с созданием временного masker

        Параметры:
            data_path: Путь к входному NIfTI файлу
            npy_path: Путь для сохранения результата (None - не сохранять)
            tr: Время повторения (Repetition Time) в секундах
            atlas_path: Путь к файлу атласа для экстракции ROI
            standardize: Флаг стандартизации данных

        Возвращает:
            np.ndarray: Массив данных формы (временные точки × ROI)
        """
        self.logger.info(f'Loading data from {data_path}')

        data_path = Path(data_path)
        atlas_path = Path(atlas_path)
        npy_path = Path(npy_path) if npy_path else None

        if not data_path.exists():
            self.logger.error(f"Data file not found: {data_path}")
            return None
        if not atlas_path.exists():
            self.logger.error(f"Atlas file not found: {atlas_path}")
            return None

        try:
            masker = NiftiLabelsMasker(
                labels_img=str(atlas_path),
                standardize=standardize,
                verbose=1,
                tr=tr
            )
            
            fmri_data = masker.fit_transform(str(data_path))
            
            if npy_path:
                npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy_path, fmri_data)
                self.logger.info(f"Saved processed data to: {npy_path}")
            
            return fmri_data
        except Exception as e:
            self.logger.error(f"Error processing {data_path}: {str(e)}", exc_info=True)
            return None


    def load_from_npy(self, npy_path):
        """
        Загрузка предобработанных данных из .npy файла

        Параметры:
        npy_path: Путь к .npy файлу

        Возвращает:
        np.ndarray: Загруженный массив данных или None при ошибке
        """
        npy_path = Path(npy_path)
        try:
            if not npy_path.exists():
                raise FileNotFoundError(f"File {npy_path} not found")
                
            self.logger.info(f"Loading preprocessed data from: {npy_path}")
            return np.load(npy_path)
        except Exception as e:
            self.logger.error(f"Error loading .npy file: {str(e)}")
            return None


    def load_events(self, events_path):
        """
        Загрузка и парсинг файла событий

        Параметры:
        events_path: Путь к текстовому файлу событий
        
        Формат файла (ожидаются как минимум 5 колонок, разделенных пробелами):
        [индекс] [onset] [длительность] [stimulus_code] [stimulus_name]

        Возвращает:
        pd.DataFrame: DataFrame с колонками:
            onset, duration, trial_type, name, stimulus_number
        """
        events_path = Path(events_path)
        if not events_path.exists():
            self.logger.error(f"Events file not found: {events_path}")
            return None
            
        return self._process_events_file(events_path)


    def load_events_for_different_trials(self, events_path):
        """
        Парсинг файла событий с группировкой по уникальным наборам вопросов.

        Логика обработки:
        - Вопросы группируются по уникальным наборам из 6 вопросов (порядок не важен).
        - Каждая группа возвращается как отдельный DataFrame.
        """
        self.logger.info(f'Parsing time-file: {events_path}')
        events_path = Path(events_path)

        # Словарь для группировки по stimulus_group (первая часть stimulus_code)
        stimulus_groups = {}
        # Словарь для связи наборов вопросов с записями
        question_groups = {}

        try:
            with events_path.open('r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split('\t')  # Разделение по табуляции
                    if len(parts) < 4:
                        continue  # Пропуск неполных строк

                    stimulus_with_question = parts[3].strip()
                    if not stimulus_with_question:
                        continue  # Пропуск строк без вопроса

                    # Разделение stimulus_code и вопроса
                    if ' ' not in stimulus_with_question:
                        continue  # Неверный формат
                    
                    stimulus_code, question = stimulus_with_question.split(' ', 1)
                    stimulus_group = stimulus_code.split('.')[0]

                    # Сохраняем вопрос в соответствующей группе
                    if stimulus_group not in stimulus_groups:
                        stimulus_groups[stimulus_group] = {
                            'questions': set(),
                            'records': []
                        }

                    # Определение параметров
                    try:
                        # stimulus_type - третий символ stimulus_code (если есть)
                        stimulus_type = stimulus_code[2] if len(stimulus_code) >= 3 else '0'
                        trial_type = 1 if stimulus_type == '4' else 0

                        stimulus_groups[stimulus_group]['questions'].add(question)
                        stimulus_groups[stimulus_group]['records'].append({
                            'onset': float(parts[1]),
                            'duration': 1.0,
                            'trial_type': trial_type,
                            'name': question,
                            'stimulus_number': stimulus_type
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")

            # Группировка по уникальным наборам вопросов
            for group, data in stimulus_groups.items():
                if len(data['questions']) != 6:
                    self.logger.warning(f"Группа {group} содержит {len(data['questions'])} вопросов. Требуется 6.")
                    continue

                # Используем frozenset для игнорирования порядка
                question_set = frozenset(data['questions'])
                if question_set not in question_groups:
                    question_groups[question_set] = []
                question_groups[question_set].extend(data['records'])

        except Exception as e:
            self.logger.error(f"Error processing events file: {str(e)}")
            return None

        # Формируем список DataFrame для каждой уникальной группы вопросов
        dfs = [pd.DataFrame(records) for records in question_groups.values()]
        return dfs if dfs else None

    def _process_events_file(self, file_path):
        """
        Парсинг файла событий с извлечением ключевых параметров

        Логика обработки:
        - stimulus_type определяется по 3-му символу stimulus_code
        - trial_type = 1 если stimulus_type == '4' (пример для конкретного эксперимента)
        """
        self.logger.info(f'Parsing time-file: {file_path}')

        records = []
        
        try:
            with file_path.open('r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    
                    if len(parts) < 5:
                        continue
                        
                    try:
                        stimulus_code = parts[3]
                        
                        stimulus_type = '0'
                        if len(stimulus_code) >= 3:
                            stimulus_type = stimulus_code[2]


                        trial_type = 1 if stimulus_type == '4' else 0

                        records.append({
                            'onset': float(parts[1]),
                            'duration': 1.0,
                            'trial_type': trial_type,
                            'name': parts[4],
                            'stimulus_number': stimulus_type
                        })
                    except Exception as e:
                        self.logger.warning(f"Error processing line {line_num}: {e}")
        except Exception as e:
            self.logger.error(f"Error processing events file: {str(e)}")
            return None
            
        return pd.DataFrame(records) if records else None

# import os
# import numpy as np
# import pandas as pd
# from nilearn.maskers import NiftiLabelsMasker
# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from nilearn.maskers import NiftiLabelsMasker
# from typing import Optional, Union



# class DataLoader:
#     # TODO: перенести создание NiftiLabelsMasker в load_from_nii_and_save, чтобы tr передавать там и не инициализировать лишний раз, если не будем скачивать из nii
#     def __init__(self, atlas_path, standartize=False, tr=1.0):
#         self.atlas_path = atlas_path
#         self.masker = NiftiLabelsMasker(labels_img=self.atlas_path, standardize=standartize, verbose=1, tr=tr)    


#     def load_data(self, data_path, npy_path=None):
#         # TODO: подумать, как поступить с этой функцией
#         '''
#             Загружаем данные в зависимости от того, есть ли данные в npy или нет
#         '''
#         if npy_path is not None:
#             return self.load_data_from_npy(npy_path)
#         else:
#             return self.load_from_nii_and_save(data_path)

    
#     def load_events(self, events_path, load_from_csv=False):
#         if load_from_csv:
#             return self.load_events_data_from_csv(events_path)
#         else:
#             return self._process_events(events_path)


#     def load_from_nii_and_save(self, data_path, npy_path=None):
#         """Загружает данные из NIfTI, преобразует и сохраняет в numpy-файл"""
        
#         # Проверяем, что файл существует
#         if not os.path.isfile(data_path):
#             print(f"Warning: File not found - {data_path}")
#             return None
        
#         try:
#             # Обработка и сохранение данных
#             fmri_data = self.masker.fit_transform(data_path)
#             if npy_path is not None:
#                 os.makedirs(os.path.dirname(npy_path), exist_ok=True)
#                 np.save(npy_path, fmri_data)   # тут надо проверить, что если нет вложенных директорий нужных
#             return fmri_data
#         except Exception as e:
#             print(f"Error processing data with path {data_path}: {str(e)}")
#             return None
    

#     def load_data_from_npy(self, npy_path):
#         """Загружает сохраненные numpy-массивы"""
#         if os.path.isfile(npy_path):
#             print(f'load from  {npy_path}')
#             return np.load(npy_path)
#         else:
#             print(f"Warning: Numpy file not found - {npy_path}")
#             return None


#     def load_events_data_from_csv(self, events_path):
#         """Загружает метки событий из CSV-файлов"""    
#         if os.path.isfile(events_path):
#             return pd.read_csv(events_path)
#         else:
#             print(f"Warning: Events file not found - {events_path}")
#             return None


#     # def process_events(self, file_name):
#     #     with open('./tmp.csv', 'w') as output_file,\
#     #         open(file_name, 'r') as input_file:
#     #             output_file.write('onset,duration,trial_type,name,stimulus_number\n')
#     #             for line in input_file:
#     #                 l = line.split()
#     #                 if len(l) == 3:
#     #                     continue
#     #                 if l[3][2] == '4':
#     #                     output_file.write(l[1] + ',1.0,'  +  '1,' + l[4] + ',' + l[3][2] + '\n')
#     #                 else:
#     #                     output_file.write(l[1] + ',1.0,' + '0,' + l[4] + ','+ l[3][2] + '\n')
#     #     events_data = pd.read_csv('./tmp.csv')

#     #     return events_data


#     def _process_events(self, file_path: Union[str, Path]) -> pd.DataFrame:
#         """Обрабатывает файл событий и возвращает DataFrame"""
#         file_path = Path(file_path)
#         records = []

#         with file_path.open('r') as input_file:
#             for line in input_file:
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue  # Пропускаем невалидные строки
                    
#                 try:
#                     # Парсим данные из строки
#                     stimulus_code = parts[3]
#                     stimulus_type = stimulus_code[2] if len(stimulus_code) >= 3 else '0'
                    
#                     record = {
#                         'onset': float(parts[1]),
#                         'duration': 1.0,
#                         'trial_type': 1 if stimulus_type == '4' else 0,
#                         'name': parts[4],
#                         'stimulus_number': stimulus_type
#                     }
#                     records.append(record)
                    
#                 except (IndexError, ValueError) as e:
#                     print(f"Error processing line: {line.strip()} - {str(e)}")
#                     continue

#         # Создаем DataFrame с явным указанием порядка колонок
#         return pd.DataFrame(records, columns=[
#             'onset', 
#             'duration', 
#             'trial_type', 
#             'name', 
#             'stimulus_number'
#         ])
