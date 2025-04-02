import os
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


class DataLoader:
    def __init__(self, atlas_path):
        self.atlas_path = atlas_path
        self.masker = NiftiLabelsMasker(labels_img=self.atlas_path, standardize=True, verbose=1)    # подумать над вот этим параметром standartize


    def load_data(self, data_path, npy_path=None):
        if npy_path is not None and os.path.isfile(npy_path):
            return self.load_data_from_npy(npy_path)
        else:
            return self.load_from_nii_and_save(data_path, npy_path)

    
    def load_events(self, events_path, load_from_csv=False):
        if load_from_csv:
            return self.load_events_data_from_csv(events_path)
        else:
            return self.process_events(events_path)

    def load_from_nii_and_save(self, data_path, npy_path=None):
        """Загружает данные из NIfTI, преобразует и сохраняет в numpy-файл"""
        
        # Проверяем, что файл существует
        if not os.path.isfile(data_path):
            print(f"Warning: File not found - {data_path}")
            return None
        
        try:
            # Обработка и сохранение данных
            fmri_data = self.masker.fit_transform(data_path)
            if npy_path is not None:
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, fmri_data)   # тут надо проверить, что если нет вложенных директорий нужных
            return fmri_data
        except Exception as e:
            print(f"Error processing data with path {data_path}: {str(e)}")
            return None
    

    def load_data_from_npy(self, data_path):
        """Загружает сохраненные numpy-массивы"""
        if os.path.isfile(data_path):
            return np.load(data_path)
        else:
            print(f"Warning: Numpy file not found - {data_path}")
            return None


    def load_events_data_from_csv(self, events_path):
        """Загружает метки событий из CSV-файлов"""    
        if os.path.isfile(events_path):
            return pd.read_csv(events_path)
        else:
            print(f"Warning: Events file not found - {events_path}")
            return None


    def process_events(self, file_name):
        with open('./tmp.csv', 'w') as output_file,\
            open(file_name, 'r') as input_file:
                output_file.write('onset,duration,trial_type\n')
                for line in input_file:
                    l = line.split()
                    if len(l) == 3:
                        continue
                    if l[3][2] == '4':
                        output_file.write(l[1] + ',1.0,'  +  '1'   + '\n')
                    else:
                        output_file.write(l[1] + ',1.0,' + '0' + '\n')
        events_data = pd.read_csv('./tmp.csv')

        return events_data

